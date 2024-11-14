import torch.onnx

from core.networks_folder.unet_binar_brats_2D_v1 import UNET_binar_brats_2D_v1
from core.load_checkpoint import *
import torch.optim as optim

def save_model(net, optimizer, epoch, loss, current_score, current_best_score, SAVE_PATH, mod):
    if mod == 'complete_save':
        current_best_score = current_best_score
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'current_best_score': current_best_score,
            }, SAVE_PATH + '\\model' + str(epoch) + '.pth')
    else:
            current_best_score = current_score
            torch.save(net.state_dict(), SAVE_PATH + '\\model' + str(epoch) + '.pth')
    return current_best_score

#Function to Convert to ONNX
def Convert_ONNX(model, MODEL_SAVE_PATH):

    # set the model to inference mode
    model.eval()

    # Let's create a dummy input tensor
    dummy_input = torch.randn(1, 4, 240, 240, requires_grad=True)

    # Export the model
    torch.onnx.export(model,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "ImageClassifier.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})

    #torch.save(model.state_dict(), MODEL_SAVE_PATH)


    #print(" ")
    #print('Model has been converted to ONNX')

MODEL_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core\models_test\model0.pth'
MODEL_SAVE_PATH = r'E:\an_4_LICENTA\Workspace\Scripturi\core\models_test\model0.onnx'

model = UNET_binar_brats_2D_v1(in_channels=4, out_channels=1, features = [8, 16, 32, ])
optimizer = optim.Adam(model.parameters(), lr=0.01)

#model.load_state_dict(torch.load(MODEL_PATH))
model, optimizer, start_epoch, loss, current_best_score = load_complete_model(model, optimizer, MODEL_PATH)


Convert_ONNX(model, MODEL_SAVE_PATH)







