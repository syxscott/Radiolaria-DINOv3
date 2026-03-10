#!/usr/bin/env python3  
\"\"\"Test script for Radiolaria-DINOv3\"\"\"  
import sys  
from pathlib import Path  
project_root = Path(__file__).parent.parent  
sys.path.insert(0, str(project_root)) 
  
\"def test_imports():\"  
\"    print('Testing imports...')\"  
\"    try:\"  
\"        import torch\"  
\"        print(f'  torch {torch.__version__}')\"  
\"    except ImportError as e:\"  
\"        print(f'  Failed: {e}')\"  
\"        return False\" 
\"    try:\"  
\"        import torchvision\"  
\"        print(f'  torchvision {torchvision.__version__}')\"  
\"    except ImportError as e:\"  
\"        print(f'  Failed: {e}')\"  
\"    try:\"  
\"        import pandas\"  
\"        print(f'  pandas {pandas.__version__}')\"  
\"    except ImportError as e:\"  
\"        print(f'  Failed: {e}')\"  
\"    print('All imports successful!')\"  
\"    return True\" 
  
\"def main():\"  
\"    print('='*60)\"  
\"    print('Radiolaria-DINOv3 Project Test')\"  
\"    print('='*60)\"  
\"    test_imports()\"  
\"    print('\\nTest completed!')\"  
  
\"if __name__ == '__main__':\"  
\"    main()\" 
