import os
import sys
import argparse

from utils import *
from train_model import *

from pre_train.sasrec.data_preprocess import preprocess
from models.a_llmrec_gnn import A_LLMRec_GNN


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # GPU train options
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument("--multi_gpu", action='store_true')
    parser.add_argument('--gpu_num', type=int, default=0)
    
    # model setting
    parser.add_argument("--llm", type=str, default='opt', help='flan_t5, opt, vicuna')
    parser.add_argument("--recsys", type=str, default='sasrec')
    
    # dataset setting
    parser.add_argument("--rec_pre_trained_data", type=str, default='Movies_and_TV')
    
    # train phase setting
    parser.add_argument("--pretrain_stage1", action='store_true')
    parser.add_argument("--pretrain_stage2", action='store_true')
    parser.add_argument("--inference", action='store_true')
    
    # hyperparameters options
    parser.add_argument('--batch_size1', default=32, type=int)
    parser.add_argument('--batch_size2', default=2, type=int)
    parser.add_argument('--batch_size_infer', default=2, type=int)
    parser.add_argument('--maxlen', default=50, type=int)
    parser.add_argument('--num_epochs', default=10, type=int)
    parser.add_argument("--stage1_lr", type=float, default=0.0001)
    parser.add_argument("--stage2_lr", type=float, default=0.0001)
    # 添加新的参数
    parser.add_argument("--use_gnn", action='store_true', help='Use GNN enhancement')
    parser.add_argument("--gnn_layers", type=int, default=3, help='Number of GNN layers')
    # 
    parser.add_argument("--use_hetero_graph", action='store_true', help='Whether to use heterogeneous graph module')
    parser.add_argument("--use_dynamic_graph", action='store_true', help='Whether to use dynamic graph updating')
    parser.add_argument("--use_contrastive", action='store_true', help='Whether to use multi-view contrastive learning')

    parser.add_argument("--experiment", type=str, default='ablation',
                       choices=['ablation', 'hyperparameter', 'cross_dataset', 'efficiency', 'all'])
    
    # ===== Checkpointing参数 =====
    parser.add_argument("--resume", action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument("--checkpoint_interval", type=int, default=200,
                       help='Save checkpoint every N steps')
    parser.add_argument("--max_checkpoints", type=int, default=3,
                       help='Maximum number of checkpoints to keep')

    # 
    parser.add_argument("--use_gradient_checkpointing", action='store_true',
                       help='Enable gradient checkpointing (saves memory)')
    
    # Stage-1 epochs（用于Stage-2加载）
    parser.add_argument("--stage1_epochs", type=int, default=10,
                       help='Stage-1 training epochs')
    args = parser.parse_args()
    
    # args.device = 'cuda:' + str(args.gpu_num)
#    args.device = 'cpu'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    if args.pretrain_stage1:
        train_model_phase1(args)
    elif args.pretrain_stage2:
        train_model_phase2(args)
    elif args.inference:
        inference(args)
