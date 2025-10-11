import os
import os.path as osp
import sys
import numpy as np

class Config:
    
    ## dataset
    # training set
    # 3D: Human36M, MuCo
    # 2D: MSCOCO, MPII 
    trainset_3d = ['Human36M']
    trainset_2d = ['MPII']

    # testing set
    # Human36M, MuPoTS, MSCOCO
    testset = 'Human36M'

    ## directory
    cur_dir = osp.dirname(os.path.abspath(__file__))
    root_dir = osp.join(cur_dir, '..')
    data_dir = osp.join(root_dir, 'data')
    output_dir = osp.join(root_dir, 'output')
    model_dir = osp.join(output_dir, 'model_dump')
    vis_dir = osp.join(output_dir, 'vis')
    log_dir = osp.join(output_dir, 'log')
    result_dir = osp.join(output_dir, 'result')
 
    ## model setting
    resnet_type = 50 # 50, 101, 152
    
    ## input, output
    input_shape = (256, 256) 
    output_shape = (input_shape[0]//8, input_shape[1]//8)
    depth_dim = 32
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)

    ## training config
    lr_dec_epoch = [17, 21]
    end_epoch = 70
    lr = 0.001 #1e-3
    lr_dec_factor = 10
    batch_size = 32
    min_lr = 1e-6
    wd = 0.1
    
    ## model variant configuration
    # Configuración por defecto (XS variant) - compatible con checkpoints antiguos
    variant = 'XS'  # Opciones: 'XS', 'S', 'M', 'L'
    backbone_cfg = ([3, 3, 9, 3], [48, 96, 192, 384])  # Default: XS
    head_cfg = None  # Se carga dinámicamente desde config_variants si se especifica variant
    
    # Legacy configurations (comentadas)
    # backbone_cfg = ([2, 2, 6, 2], [40, 80, 160, 320])  # Otra configuración
    
    depth = 256
    warmup_epochs = 5
    
    @staticmethod
    def load_variant_config(variant_name):
        """
        Carga la configuración de backbone y head para una variante específica.
        
        Args:
            variant_name (str): Nombre de la variante ('XS', 'S', 'M', 'L')
        
        Modifica:
            cfg.variant, cfg.backbone_cfg, cfg.head_cfg
        
        Ejemplo:
            Config.load_variant_config('M')  # Carga configuración para ConvNeXtPose-M
        """
        try:
            from config_variants import get_model_config, get_full_config
            
            # Cargar configuración de backbone
            depths, dims = get_model_config(variant_name)
            Config.backbone_cfg = (depths, dims)
            
            # Cargar configuración completa (incluye head_cfg)
            full_config = get_full_config(variant_name)
            Config.head_cfg = full_config.get('head_cfg', None)
            Config.variant = variant_name
            
            print(f"✓ Configuración cargada para variante: {variant_name}")
            print(f"  - Backbone: depths={depths}, dims={dims}")
            if Config.head_cfg:
                print(f"  - HeadNet: {Config.head_cfg['num_deconv_layers']}-UP "
                      f"({Config.head_cfg['num_deconv_layers']} capas de upsampling)")
            
        except ImportError:
            print(f"⚠️  Advertencia: config_variants.py no encontrado.")
            print(f"   Usando configuración legacy por defecto.")
        except Exception as e:
            print(f"❌ Error cargando variante '{variant_name}': {e}")
            print(f"   Usando configuración legacy por defecto.")
    save_interval = 1

    ## testing config
    test_batch_size = 16
    flip_test = True
    use_gt_info = True

    ## others
    num_thread = 20
    gpu_ids = '0'
    num_gpus = 1
    continue_train = False

    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))

cfg = Config()

sys.path.insert(0, osp.join(cfg.root_dir, 'common'))
from utils.dir_utils import add_pypath, make_folder
add_pypath(osp.join(cfg.data_dir))
for i in range(len(cfg.trainset_3d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_3d[i]))
for i in range(len(cfg.trainset_2d)):
    add_pypath(osp.join(cfg.data_dir, cfg.trainset_2d[i]))
add_pypath(osp.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)

