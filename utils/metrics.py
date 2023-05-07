import torch
import torch.distributed as dist

class MetricTracker():
    """
    원하는 Metric을 전부 계산하고 Dict형태로 출력하는 클래스
    - Train시 Score는 첫 번쨰 Interval부터 축척된 평균값임
    - Resume시 checkpoint까지 저장된 평균값에서부터 다시 시작
    - Resume시 self.sum = checkpoint['avg']
    """
    def __init__(self, opt):
        self.opt = opt
        self.world_size = opt.WORLD_SIZE
        self.metrics = opt.CHECKPOINT.METRICS
        
        # Resume모드일 땐 기존 avg를 이어서 사용하기 때문에 interval=1 로 시작
        self.interval = 0 if self.opt.MODEL.RESUME_PATH is None else 1
        self.sum = {m:0 for m in opt.CHECKPOINT.METRICS}
        self.avg = {m:0 for m in opt.CHECKPOINT.METRICS}

        
    def get(self, output, target, rank):
        """
        1 interval 당 Metric값 계산 (Batch 및 Multi-GPU 평균)

        output: [B, H, W] or [B, 1, H, W]
        target: [B, H, W] or [B, 1, H, W]

        return: {
            'metric1': score1,
            'metric2': score2
        }
        """
        self.interval += 1

        pred = (torch.sigmoid(output[-1].cpu()) > self.opt.CHECKPOINT.THRESHOLD).float()
        for metric in self.metrics:
            score = eval(metric)(pred, target).to(rank) # 각 Metric별 score 계산
            dist.all_reduce(score, op=dist.ReduceOp.SUM) # Multi-GPU 결과 합산

            self.sum[metric] += (score / self.opt.WORLD_SIZE).item()
            self.avg[metric] = self.sum[metric] / self.interval

        return self.avg


# # https://github.com/pytorch/pytorch/issues/1249
def Dice(input, target):
    num_in_target = input.shape[0]
    
    smooth = 1.

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    intersection = (pred * truth).sum(1)

    score = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)

    return score.mean()

def Dice2(input, target):
    num_in_target = input.shape[0]
    
    smooth = 1.

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    pred = torch.where(pred == 1, 1, 0)
    truth = torch.where(truth == 1, 1, 0)

    intersection = (pred * truth).sum(1)

    score = (2. * intersection + smooth) /(pred.sum(1) + truth.sum(1) + smooth)

    return score.mean()

def IoU(input, target):
    num_in_target = input.size(0)

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    # intersection = (pred*truth).long().sum(1).data.cpu()[0]
    intersection = (pred * truth).sum(1)
    
    # union = input.long().sum().data.cpu()[0] + target.long().sum().data.cpu()[0] - intersection
    union = pred.sum(1) + truth.sum(1) - intersection

    score = (intersection + 1e-15) / (union + 1e-15)

    return score.mean()

def IoU2(input, target):
    num_in_target = input.size(0)

    pred = input.view(num_in_target, -1)
    truth = target.view(num_in_target, -1)

    pred = torch.where(pred == 1, 1, 0)
    truth = torch.where(truth == 1, 1, 0)

    # intersection = (pred*truth).long().sum(1).data.cpu()[0]
    intersection = (pred * truth).sum(1)
    
    # union = input.long().sum().data.cpu()[0] + target.long().sum().data.cpu()[0] - intersection
    union = pred.sum(1) + truth.sum(1) - intersection

    score = (intersection + 1e-15) / (union + 1e-15)

    return score.mean()