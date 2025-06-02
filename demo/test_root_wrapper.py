from root_wrapper import RootNetWrapper

root_wrapper = RootNetWrapper('/home/fabri/3DMPPE_ROOTNET_RELEASE', '/home/fabri/3DMPPE_ROOTNET_RELEASE/demo/snapshot_18.pth.tar')
root_wrapper.load_model(use_gpu=False)
