import numpy as np
import torch
import torch.nn as nn

'''https://blog.naver.com/kimnanhee97/222053358821 참고'''

class SegmentationMetrics(object):
    r"""Calculate common metrics in semantic segmentation to evalueate model preformance.
    Supported metrics: Pixel accuracy, Dice Coeff, precision score and recall score.
    
    Pixel accuracy measures how many pixels in a image are predicted correctly.
    Dice Coeff is a measure function to measure similarity over 2 sets, which is usually used to
    calculate the similarity of two samples. Dice equals to f1 score in semantic segmentation tasks.
    
    It should be noted that Dice Coeff and Intersection over Union are highly related, so you need 
    NOT calculate these metrics both, the other can be calcultaed directly when knowing one of them.
    Precision describes the purity of our positive detections relative to the ground truth. Of all
    the objects that we predicted in a given image, precision score describes how many of those objects
    actually had a matching ground truth annotation.
    Recall describes the completeness of our positive predictions relative to the ground truth. Of
    all the objected annotated in our ground truth, recall score describes how many true positive instances
    we have captured in semantic segmentation.
    Args:
        eps: float, a value added to the denominator for numerical stability.
            Default: 1e-5
        average: bool. Default: ``True``
            When set to ``True``, average Dice Coeff, precision and recall are
            returned. Otherwise Dice Coeff, precision and recall of each class
            will be returned as a numpy array.
        ignore_background: bool. Default: ``True``
            When set to ``True``, the class will not calculate related metrics on
            background pixels. When the segmentation of background pixels is not
            important, set this value to ``True``.
        activation: [None, 'none', 'softmax' (default), 'sigmoid', '0-1']
            This parameter determines what kind of activation function that will be
            applied on model output.
    Input:
        y_true: :math:`(N, H, W)`, torch tensor, where we use int value between (0, num_class - 1)
        to denote every class, where ``0`` denotes background class.
        y_pred: :math:`(N, C, H, W)`, torch tensor.
    Examples::
        >>> metric_calculator = SegmentationMetrics(average=True, ignore_background=True)
        >>> pixel_accuracy, dice, precision, recall = metric_calculator(y_true, y_pred)
    """
    def __init__(self, eps=1e-5, average=True, ignore_background=True, activation='0-1'):
        self.eps = eps
        self.average = average
        self.ignore = ignore_background
        self.activation = activation

    @staticmethod
    def _one_hot(gt, pred, class_num):
        # transform sparse mask into one-hot mask
        # shape: (B, H, W) -> (B, C, H, W)
        input_shape = tuple(gt.shape)  # (N, H, W, ...)
        new_shape = (input_shape[0], class_num) + input_shape[1:]
        one_hot = torch.zeros(new_shape).to(pred.device, dtype=torch.float)
        target = one_hot.scatter_(1, gt.unsqueeze(1).long().data, 1.0)
        return target

    @staticmethod
    def _get_class_data(gt_onehot, pred, class_num):
        # perform calculation on a batch
        # for precise result in a single image, plz set batch size to 1
        matrix = np.zeros((3, class_num))

        # calculate tp, fp, fn per class
        for i in range(class_num):
            # pred shape: (N, H, W)
            class_pred = pred[:, i, :, :]
            # gt shape: (N, H, W), binary array where 0 denotes negative and 1 denotes positive
            class_gt = gt_onehot[:, i, :, :]

            pred_flat = class_pred.contiguous().view(-1, )  # shape: (N * H * W, )
            gt_flat = class_gt.contiguous().view(-1, )  # shape: (N * H * W, )

            tp = torch.sum(gt_flat * pred_flat)
            fp = torch.sum(pred_flat) - tp
            fn = torch.sum(gt_flat) - tp

            matrix[:, i] = tp.item(), fp.item(), fn.item()

        return matrix

    def _calculate_multi_metrics(self, gt, pred, class_num):
        # calculate metrics in multi-class segmentation
        matrix = self._get_class_data(gt, pred, class_num)
        if self.ignore:
            matrix = matrix[:, 1:]

        # tp = np.sum(matrix[0, :])
        # fp = np.sum(matrix[1, :])
        # fn = np.sum(matrix[2, :])

        pixel_acc = (np.sum(matrix[0, :]) + self.eps) / (np.sum(matrix[0, :]) + np.sum(matrix[1, :]))
        dice = (2 * matrix[0] + self.eps) / (2 * matrix[0] + matrix[1] + matrix[2] + self.eps)
        precision = (matrix[0] + self.eps) / (matrix[0] + matrix[1] + self.eps)
        recall = (matrix[0] + self.eps) / (matrix[0] + matrix[2] + self.eps)
        f1 = 2 * ((precision * recall) / (precision + recall))
        # iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        
        
        if self.average:
            dice = np.average(dice)
            precision = np.average(precision)
            recall = np.average(recall)
            f1 = np.average(f1)

        return precision, recall, f1, pixel_acc, dice, 

    def __call__(self, y_true, y_pred):
        class_num = y_pred.size(1)

        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "softmax":
            activation_fn = nn.Softmax(dim=1)
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            pred_argmax = torch.argmax(y_pred, dim=1)
            activated_pred = self._one_hot(pred_argmax, y_pred, class_num)
        else:
            raise NotImplementedError("Not a supported activation!")

        gt_onehot = self._one_hot(y_true, y_pred, class_num)
        precision, recall, f1, pixel_acc, dice = self._calculate_multi_metrics(gt_onehot, activated_pred, class_num)
        return precision, recall, f1, pixel_acc, dice


class BinaryMetrics():
    r"""Calculate common metrics in binary cases.
    In binary cases it should be noted that y_pred shape shall be like (N, 1, H, W), or an assertion 
    error will be raised.
    Also this calculator provides the function to calculate specificity, also known as true negative 
    rate, as specificity/TPR is meaningless in multiclass cases.
    """
    def __init__(self, eps=1e-5, activation='0-1'):
        self.eps = eps
        self.activation = activation

    def _calculate_overlap_metrics(self, gt, pred):
        output = pred.view(-1, )
        target = gt.view(-1, ).float()

        tp = torch.sum(output * target)  # TP
        fp = torch.sum(output * (1 - target))  # FP
        fn = torch.sum((1 - output) * target)  # FN
        tn = torch.sum((1 - output) * (1 - target))  # TN

        pixel_acc = (tp + tn + self.eps) / (tp + tn + fp + fn + self.eps)
        dice = (2 * tp + self.eps) / (2 * tp + fp + fn + self.eps)
        precision = (tp + self.eps) / (tp + fp + self.eps)
        recall = (tp + self.eps) / (tp + fn + self.eps)
        specificity = (tn + self.eps) / (tn + fp + self.eps)
        iou = (tp + self.eps) / (tp + fp + fn + self.eps)
        f1 = 2 * ((precision * recall) / (precision + recall))
        
        return precision, recall, f1, iou, pixel_acc, dice, specificity

    def __call__(self, y_true, y_pred):
        # y_true: (N, H, W)
        # y_pred: (N, 1, H, W)
        if self.activation in [None, 'none']:
            activation_fn = lambda x: x
            activated_pred = activation_fn(y_pred)
        elif self.activation == "sigmoid":
            activation_fn = nn.Sigmoid()
            activated_pred = activation_fn(y_pred)
        elif self.activation == "0-1":
            sigmoid_pred = nn.Sigmoid()(y_pred)
            activated_pred = (sigmoid_pred > 0.5).float().to(y_pred.device)
        else:
            raise NotImplementedError("Not a supported activation!")

        assert activated_pred.shape[1] == 1, 'Predictions must contain only one channel' \
                                             ' when performing binary segmentation'
        precision, recall, f1, iou, pixel_acc, dice, specificity = self._calculate_overlap_metrics(y_true.to(y_pred.device,
                                                                                                    dtype=torch.float),
                                                                                          activated_pred)
        return [precision, recall, f1, iou, pixel_acc, dice, specificity]
    

def print_metrics(label, output, idx, average=False):
    metric_cal = BinaryMetrics()
    precision, recall, f1, iou, pixel_acc, _, _ = metric_cal(label, output)
    metric_results = {
        'pixel_acc': pixel_acc,
        'precision': precision,
        'recall': recall,
        'f1-score': f1,
        'IOU': iou,
    }
    msg = ''
    if average:
        for k, v in metric_results.items():
            msg.join(f'{k}: ')
    
if __name__ == '__main__':
    import albumentations as A
    from torch.utils.data import DataLoader
    from albumentations.pytorch import ToTensorV2
    import numpy as np
    from dataset import DeepglobeDataset, DeepglobeDatasetMulticlass, LoveDADataset
    import torch
    transform_train = A.Compose([
        A.Rotate(limit=40,p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        ToTensorV2()
    ])
    
    img_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_image'
    # mask_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_segment'
    mask_dir = '/home/yh.sakong/data/deepglobe_road/deepglobe512/train_landcover'
    
    # trainset = DeepglobeDataset(img_dir, mask_dir, transform_train)
    trainset = LoveDADataset(img_dir, mask_dir, transform_train)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
    print(len(trainset))
    
    img, mask = iter(trainloader).next()
    print(np.unique(mask))
    METRIC_AVERAGE = False
    # cal = BinaryMetrics()
    cal = SegmentationMetrics(average=True, ignore_background=False)
    with torch.no_grad():
        from networks import create_model
        
        model = create_model(model_name='segformer', num_classes=8)
        output = model(img)
    print('mask shape', mask.shape)
    print('output shape', output.shape)
    
    metric_sum = 0
    metric_results = np.array(cal(mask.cpu(), output.cpu())) # label : (N, H, W) / output : (N, 1, H, W)
    print(metric_results)
    # metric_sum += metric_results
    # msg = f'Train ({10}) | '
    # if METRIC_AVERAGE:
    #     metric_avg = metric_sum / (1)
    #     for i, metric in enumerate(['Pre', 'Recall', 'F1', 'IOU', 'Pixel_acc']):
    #         msg += f'{metric}: {metric_avg[i]} | '
    # else:
    #     for i, metric in enumerate(['Pre', 'Recall', 'F1', 'IOU', 'Pixel_acc']):
    #         msg += f'{metric}: {metric_results[i]:0.4f} | '
            
    # print(msg)
    
    '''        
    y_true: :math:`(N, H, W)`, torch tensor, where we use int value between (0, num_class - 1)
    to denote every class, where ``0`` denotes background class.
    y_pred: :math:`(N, C, H, W)`, torch tensor.
        
    '''