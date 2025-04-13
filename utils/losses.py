import torch

def SILogLoss(output, target, lmbda=0.5, is_output_logarithm = False):
    """
    loss defined here: https://torch-uncertainty.github.io/generated/torch_uncertainty.metrics.regression.SILog.html
    
    target,output shape: (B, C, H, W) = (B, 1, H, W)
    - Batch B
    - Channels C = 1 (may also be already squeezed)
    - Heigth H
    - Width W

    lmbda regularization parameter.
    """
    assert output.shape[0] == target.shape[0], f"Batch Size should be the same between output ({output.shape=}) and target tensors ({target.shape=})." 
    assert output.shape[-1] == target.shape[-1], f"Image Width should be the same between output ({output.shape=}) and target tensors ({target.shape=})." 
    assert output.shape[-2] == target.shape[-2], f"Image Height should be the same between output ({output.shape=}) and target tensors ({target.shape=})." 

    output = output if output.shape[1] != 1 else output.squeeze(1)
    target = target if target.shape[1] != 1 else target.squeeze(1)

    output_logits = torch.log(output) if not is_output_logarithm else output
    target_logits = torch.log(target)

    delta_log_diff = target_logits - output_logits
    num_elements = delta_log_diff[0].numel()

    term1 = torch.square(delta_log_diff)
    term1 = torch.mean(term1)

    term2 = torch.sum(delta_log_diff, dim=(1,2))/num_elements
    term2 = torch.square(term2)
    term2 = torch.mean(term2)
    
    Loss_average = term1 - lmbda*term2
    return Loss_average

def GradientLoss(output, target, visualization=False):
    """

    target,output shape: (B, C, H, W) = (B, 1, H, W)
    - Batch B
    - Channels C = 1
    - Heigth H
    - Width W

    computes:
    - spatial gradients in y and x directions for both the target and the output.
    - gradient magnitude at each pixel (similar to edge strength in edge detection).
    - Zeroes out small gradients to suppress weak edges.
    - the difference in gradient magnitudes (edges) between predicted and target
    - Averages over the entire batch and images.
    """
    dy_targ, dx_targ = torch.gradient(target, dim=(2,3))
    dy_out, dx_out = torch.gradient(output, dim=(2,3))
    
    dy_targ = dy_targ[:, 0, :, :]
    dx_targ = dx_targ[:, 0, :, :]
    dy_out = dy_out[:, 0, :, :]
    dx_out = dx_out[:, 0, :, :]

    new_image_tensor = (dx_targ)**2+(dy_targ)**2
    new_image_tensor = torch.sqrt(new_image_tensor)
    output_sim = (dx_out)**2+(dy_out)**2
    output_sim = torch.sqrt(output_sim)

    canny_edge_detector_target = torch.where(new_image_tensor < 0.020, torch.tensor(0.0), new_image_tensor)

    if visualization:
        import cv2
        import numpy as np
        
        img1 = np.uint8(canny_edge_detector_target[0].detach().cpu().numpy()*255)
        img2 = output[0][0].detach().cpu().numpy()
        img2 = np.uint8(cv2.normalize(img2, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_64F))
        img3 = target[0][0].detach().cpu().numpy()
        img3 = np.uint8(cv2.normalize(img3, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_64F))

        side_by_side = np.hstack([img1, img2, img3])
        cv2.imshow('Side by Side', side_by_side)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    diff_edge = torch.square(output_sim-canny_edge_detector_target)
    diff_edge = torch.sqrt(diff_edge)
    return torch.mean(diff_edge)
    # diff_y = (dy_targ-dy_out)**2
    # diff_x = (dx_targ-dx_out)**2

    # diff_x_mean = torch.mean(diff_x)
    # diff_y_mean = torch.mean(diff_y)


def SIRMSELoss(output, target, check_shape=False, is_output_logarithm=False):
    """
    Scale-Invariant RSME Loss. more info: https://www.kaggle.com/competitions/ethz-cil-monocular-depth-estimation-2025/overview

    target,output shape: (B, C, H, W) = (B, 1, H, W)
    - Batch B
    - Channels C = 1 (may also be already squeezed)
    - Heigth H
    - Width W
    """
    assert output.shape[0] == target.shape[0], "Batch Size should be the same between output and target tensors." 
    assert output.shape[-1] == target.shape[-1], "Image Width should be the same between output and target tensors." 
    assert output.shape[-2] == target.shape[-2], "Image Height should be the same between output and target tensors." 

    output = output if output.shape[1] != 1 else output.squeeze(1)
    target = target if target.shape[1] != 1 else target.squeeze(1)

    output_logits = torch.log(output) if not is_output_logarithm else output
    target_logits = torch.log(target)

    delta_log_diff = output_logits - target_logits
    alpha = torch.mean(-delta_log_diff, dim=[1,2], keepdim=True)

    summand = torch.square(delta_log_diff + alpha)
    batch_loss = torch.sqrt(torch.mean(summand, dim=[1,2]))

    loss = torch.mean(batch_loss)
    
    return loss


if __name__ == '__main__':
    # check correctness of each loss

    T = torch.zeros((2,4,5))
    print(f"{T.shape = }")
    print(f"{T[1][0] = }")

    A = 1 + T
    print(f"{SILogLoss(torch.exp(A), torch.exp(T)) = }") 
    # Notice term1 = 1, term2 = 1. With lmbda = 0.5 we recover the result


    print(f"{SIRMSELoss(torch.exp(A), torch.exp(T)) = }")
    # Notice alpha = -1, delta = 1, loss=0


    import cv2
    import numpy as np
    from utils import random_image, random_mask

    img = np.expand_dims(np.array([random_mask(), random_mask()]), axis=1)
    img1 = img / 2
    img2 = img / 2 + np.uint8(np.random.rand(*img.shape)*3)/2

    print(f"{GradientLoss(torch.tensor(img), torch.tensor(img), True) = }") # for equal images the loss is not 0? wtf
    print(f"{GradientLoss(torch.tensor(img1), torch.tensor(img), True) = }")
    print(f"{GradientLoss(torch.tensor(img), torch.tensor(img1), True) = }")
    print(f"{GradientLoss(torch.tensor(img2), torch.tensor(img), True) = }")
    print(f"{GradientLoss(torch.tensor(img), torch.tensor(img2), True) = }")




