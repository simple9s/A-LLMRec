from pytz import timezone
import torch
import os
import gc
import json
import psutil
import GPUtil
from datetime import datetime

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# ex. target_word: .csv / in target_path find 123.csv file
def find_filepath(target_path, target_word):
    file_paths = []
    for file in os.listdir(target_path):
        if os.path.isfile(os.path.join(target_path, file)):
            if target_word in file:
                file_paths.append(target_path + file)
            
    return file_paths

def create_file_if_not_exists(file_path):
    # 获取目录路径
    directory = os.path.dirname(file_path)
    
    # 如果目录不存在，创建目录（包括多级目录）
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        print(f"创建目录: {directory}")
    
    # 如果文件不存在，创建文件
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass  # 创建空文件
        print(f"创建文件: {file_path}")
        return False
    else:
        print(f"文件已存在: {file_path}")
        return True

    
    
    

class CheckpointManager:
    """Checkpoint管理器 - 支持断点续训"""
    
    def __init__(self, checkpoint_dir='./checkpoints', max_keep=3):
        self.checkpoint_dir = checkpoint_dir
        self.max_keep = max_keep
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def save_checkpoint(self, model, optimizer, epoch, step, stage, 
                       loss=None, metrics=None, args=None):
        """保存checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'stage': stage,
            'model_state_dict': model.state_dict() if not hasattr(model, 'module') else model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'args': vars(args) if args else None,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 文件名包含stage、epoch、step
        filename = f'checkpoint_stage{stage}_epoch{epoch}_step{step}.pt'
        filepath = os.path.join(self.checkpoint_dir, filename)
        
        # 保存
        torch.save(checkpoint, filepath)
        print(f"✓ Checkpoint saved: {filepath}")
        
        # 清理旧checkpoint
        self._cleanup_old_checkpoints(stage)
        
        return filepath
    
    def load_checkpoint(self, model, optimizer=None, checkpoint_path=None, 
                       stage=None, device='cuda'):
        """加载checkpoint"""
        # 如果没有指定路径，查找最新的
        if checkpoint_path is None:
            checkpoint_path = self.find_latest_checkpoint(stage)
        
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print(f"No checkpoint found at {checkpoint_path}")
            return None
        
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型状态
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"✓ Loaded checkpoint: Epoch {checkpoint['epoch']}, Step {checkpoint['step']}")
        
        return checkpoint
    
    def find_latest_checkpoint(self, stage=None):
        """查找最新的checkpoint"""
        if not os.path.exists(self.checkpoint_dir):
            return None
        
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pt'):
                if stage is None or f'stage{stage}' in filename:
                    filepath = os.path.join(self.checkpoint_dir, filename)
                    checkpoints.append((filepath, os.path.getmtime(filepath)))
        
        if not checkpoints:
            return None
        
        # 按修改时间排序，返回最新的
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        return checkpoints[0][0]
    
    def _cleanup_old_checkpoints(self, stage):
        """清理旧的checkpoint，只保留最新的max_keep个"""
        checkpoints = []
        for filename in os.listdir(self.checkpoint_dir):
            if filename.endswith('.pt') and f'stage{stage}' in filename:
                filepath = os.path.join(self.checkpoint_dir, filename)
                checkpoints.append((filepath, os.path.getmtime(filepath)))
        
        # 按时间排序
        checkpoints.sort(key=lambda x: x[1], reverse=True)
        
        # 删除多余的
        for filepath, _ in checkpoints[self.max_keep:]:
            os.remove(filepath)
            print(f"Removed old checkpoint: {filepath}")


class GradientCheckpointing:
    """梯度检查点 - 用于Stage-2的LLM训练"""
    
    @staticmethod
    def enable_gradient_checkpointing(model):
        """启用梯度检查点"""
        if hasattr(model, 'llm') and hasattr(model.llm, 'llm_model'):
            llm_model = model.llm.llm_model
            
            # OPT模型的gradient checkpointing
            if hasattr(llm_model, 'gradient_checkpointing_enable'):
                llm_model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled for LLM")
            
            # 设置use_cache=False（gradient checkpointing需要）
            if hasattr(llm_model.config, 'use_cache'):
                llm_model.config.use_cache = False
    
    @staticmethod
    def disable_gradient_checkpointing(model):
        """禁用梯度检查点"""
        if hasattr(model, 'llm') and hasattr(model.llm, 'llm_model'):
            llm_model = model.llm.llm_model
            
            if hasattr(llm_model, 'gradient_checkpointing_disable'):
                llm_model.gradient_checkpointing_disable()
                print("✓ Gradient checkpointing disabled")
            
            if hasattr(llm_model.config, 'use_cache'):
                llm_model.config.use_cache = True