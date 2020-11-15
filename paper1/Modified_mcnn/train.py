from model.hetero.mcnn import Hetero,Modified_m_CNN
from dataset.data_generator import OxfordFlower
import argparse

import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark_dataset', type=str, default='omniglot')
    parser.add_argument('--network_cls', type=str, default='omniglot')
    parser.add_argument('--n', type=int, default=5) # 필요없음
    parser.add_argument('--epochs', type=int, default=5) # 중복됌
    parser.add_argument('--iterations', type=int, default=5) # 필요없음
    parser.add_argument('--k', type=int, default=1) # 필요없음
    parser.add_argument('--meta_batch_size', type=int, default=2)
    parser.add_argument('--num_steps_ml', type=int, default=10)
    parser.add_argument('--lr_inner_ml', type=float, default=0.4)
    parser.add_argument('--num_steps_validation', type=int, default=10)
    # parser.add_argument('--save_after_epochs', type=int, default=500)
    parser.add_argument('--save_after_epochs', type=int, default=1)
    parser.add_argument('--meta_learning_rate', type=float, default=0.001)
    parser.add_argument('--report_validation_frequency', type=int, default=50)
    parser.add_argument('--log_train_images_after_iteration', type=int, default=1)
    
    # MultiModal Argument
    parser.add_argument("--batch_size",type=int,default=64)
    parser.add_argument("--batch_size2",type=int,default=64)# image encoder을 pretrain시킬때의 batch_size
    #parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--epochs2",type=int,default=5) # image encoder을 pretrain시킬때의 epochs

    parser.add_argument("--encoder_name",type=str,default="MobileNetV2")
    parser.add_argument("--optimizer",type=str,default="SGD")

    parser.add_argument("--binary",type=bool,default=False) # is binary classification?
    parser.add_argument("--pretrain",type=bool,default=True) # does pretrain 
    parser.add_argument("--fineTune",type=bool,default=True)
    
    args = parser.parse_args()

    # 데이터셋 객체를 생성합니다.
    # 타입 : tf.data.Dataset
    if args.benchmark_dataset == "omniglot":
        database = OmniglotDatabase(
            raw_data_address="dataset\omniglot",
            random_seed=47,
            num_train_classes=1200,
            num_val_classes=100)
    elif args.benchmark_dataset == "mini_imagenet":
        database=MiniImagenetDatabase(
            raw_data_address="\dataset\mini_imagenet",
            random_seed=-1)
    elif args.benchmark_dataset == "oxford_flower":
        database = OxfordFlower(
            config_path="./dataset/raw_data/oxfordflower/args.ini",
            random_seed=47)
            

    # 모델 객체를 생성합니다.
    if args.network_cls == "omniglot":
        network_cls=OmniglotModel
    elif args.network_cls == "mini_imagenet":
        network_cls=MiniImagenetModel
    elif args.network_cls == "modified_mcnn":
        network_cls=Modified_m_CNN
    
    if network_cls in [OmniglotModel,MiniImagenetModel]:
        maml = ModelAgnosticMetaLearning(args, database, network_cls)
        maml.meta_train(epochs = args.epochs)
    elif network_cls in [Modified_m_CNN]:
        hetero = Hetero(args,"./dataset/raw_data/oxfordflower/args.ini",database,network_cls)
        hetero.train()
        hetero.evaluate()
