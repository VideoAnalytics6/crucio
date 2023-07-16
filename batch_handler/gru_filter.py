import torch
import torch.nn as nn
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights, mobilenet_v2
from torchvision.models.mobilenetv3 import (MobileNet_V3_Small_Weights,
                                            mobilenet_v3_small)
from torchvision.models.squeezenet import (SqueezeNet1_0_Weights,
                                           SqueezeNet1_1_Weights,
                                           squeezenet1_0, squeezenet1_1)

from autoencoder.dataset import MIN_FRAME_NUM
from autoencoder.util import DOWNLOAD_DIR, GPU_ENABLED

INPUT_SIZE = 1000
HIDDEN_SIZE = 256
LAYER_NUM = 1
GRU_PATH = DOWNLOAD_DIR+'/weights_gru_' + \
    str(INPUT_SIZE)+'.'+str(HIDDEN_SIZE)+'.'+str(LAYER_NUM)+'.pth'
print('Network parameters of GRU (Beneficial to keyframe extraction but longer filtering time)')
print(f'INPUT_SIZE={INPUT_SIZE} (Must be same size as VideoExtractor output)')
print(f'HIDDEN_SIZE={HIDDEN_SIZE}')
print(f'LAYER_NUM={LAYER_NUM}')
print(f'GRU_PATH={GRU_PATH}')


class VideoExtractor(nn.Module):
    def __init__(self):
        super(VideoExtractor, self).__init__()
        self.extractor = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.extractor = mobilenet_v3_small(
            weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.extractor = squeezenet1_0(weights=SqueezeNet1_0_Weights.DEFAULT)
        self.extractor = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        self.extractor.eval()

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[1], shape[-2], shape[-1])
        x = self.extractor(x)
        features = x.reshape(shape[0], shape[2], -1)
        size = x.shape[-1]
        return features, size


class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(INPUT_SIZE, HIDDEN_SIZE, LAYER_NUM,
                          batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2*HIDDEN_SIZE, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        self.gru.flatten_parameters()
        features, _ = self.gru(x)
        scores = self.fc(features).squeeze()
        scores = scores.reshape(x.shape[0], -1)
        scores = self.sig(scores)
        return scores


def get_filter(mode='train', is_load=False):
    '''
    mode -> Set network mode to 'train' or 'eval'
    is_load -> Whether to load GRU model weights 
               for continued training or evaluation
    '''
    extractor = VideoExtractor()
    extractor = extractor.cuda()
    gru = GRUModel()
    if mode == 'train':
        gru = gru.cuda()
        gru.train()
    elif mode == 'eval':
        if GPU_ENABLED:
            gru = gru.cuda()
        gru.eval()
    if is_load:
        gru.load_state_dict(torch.load(GRU_PATH))
    return extractor, gru


def length_regularization(selects, percentage=0.5):
    '''
    Calculate length regression of current selection
    '''
    length = selects.sum(dim=1).mean()
    return torch.abs(length/selects.shape[1]-percentage)


def diversity_regularization(features, selects, inf=1e-4):
    '''
    Calculate diversity regression of current selection
    '''
    assert features.shape[0] == selects.shape[0]
    batch_size = features.shape[0]
    dpp_loss = 0
    for _ in range(batch_size):
        N = selects[_]
        assert torch.all(N == 0) == False
        X = features[_]
        # Similarity matrix L is calculated based on Gaussian kernel function
        sigma = 1
        L = torch.exp(-torch.cdist(X, X, p=2) ** 2 / (2 * sigma ** 2))
        # Compute similarity of selected subset
        # ?torch.nonzero causes gradient function of selected to disappear
        selected = torch.nonzero(N).squeeze()
        L_s = L.index_select(0, selected).index_select(1, selected)
        det_L_s = torch.det(L_s)
        assert det_L_s >= 0
        # Calculate DPP probability
        I = torch.eye(X.size(0), requires_grad=True).cuda()
        det_L = torch.det(L + I)
        dpp = (det_L_s+inf) / det_L
        assert dpp <= 1
        prob = -torch.log(dpp)
        dpp_loss += prob
    dpp_loss /= batch_size
    return dpp_loss


def test_diversity_regularization():
    feature = torch.tensor(
        [[[1, 2, 3], [1, 2, 3], [2, 3, 4], [5, 6, 7]],
         [[3, 2, 1], [3, 2, 1], [3, 4, 5], [3, 4, 5]]],
        dtype=torch.float32).cuda()
    selects1 = torch.ones(2, 4, dtype=torch.float32).cuda()
    selects2 = torch.tensor([[0, 1, 1, 1], [1, 0, 1, 1]],
                            dtype=torch.float32).cuda()
    selects3 = torch.tensor([[1, 0, 1, 1], [1, 0, 1, 0]],
                            dtype=torch.float32).cuda()
    print(diversity_regularization(feature, selects1))
    print(diversity_regularization(feature, selects2))
    print(diversity_regularization(feature, selects3))


def representativeness_loss(criterion, videos, selects):
    '''
    Representativeness loss of current selection is calculated based on loss function criterion
    '''
    assert videos.shape[0] == selects.shape[0]
    batch_size = videos.shape[0]
    rep_loss = 0
    for _ in range(batch_size):
        N = selects[_]
        assert torch.all(N == 0) == False
        X = videos[_]
        X = X.transpose(0, 1)
        number = N.shape[0]
        rep_mat = torch.zeros(number, number, requires_grad=True).cuda()
        for i in range(number):
            for j in range(number):
                rep_mat[i][j] = criterion(
                    X[i].unsqueeze(0), X[j].unsqueeze(0))
        # ?torch.nonzero causes gradient function of indices to disappear
        indices = torch.nonzero(N).squeeze()
        rep_mat = rep_mat[:, indices].reshape(number, -1)
        rep_mat = torch.exp(rep_mat.min(dim=1, keepdim=True)[0])
        rep_loss += rep_mat.mean()
    rep_loss /= batch_size
    return rep_loss


def scores_to_selects(scores, norm=False):
    batch_size = scores.shape[0]
    normalized_scores = torch.zeros(scores.shape, requires_grad=True).cuda()
    for _ in range(batch_size):
        score = scores[_]
        normalized_scores[_] = (score - score.min()) / \
            (score.max() - score.min())
    if norm:
        return normalized_scores
    else:
        selects = torch.round(normalized_scores)
        # Adjust selects according to MIN_FRAME_NUM
        for _ in range(batch_size):
            ones = (selects[_] == 1).sum().item()
            if ones < MIN_FRAME_NUM:
                count = MIN_FRAME_NUM - ones
                indices = torch.where(normalized_scores[_] < 0.5)[0]
                top_indices = torch.topk(
                    normalized_scores[_][indices], k=count).indices
                result_indices = indices[top_indices]
                selects[_][result_indices] = 1
        return selects


def apply_select_to_video(selects, videos):
    length = len(selects)
    selects = torch.nonzero(selects)
    select_videos = []
    for _ in range(length):
        idx = (selects[:, 0] == _).nonzero().squeeze()
        select = selects[idx, 1]
        video = videos[_]
        select_video = torch.index_select(video, dim=1, index=select)
    select_videos.append(select_video)
    return select_videos
