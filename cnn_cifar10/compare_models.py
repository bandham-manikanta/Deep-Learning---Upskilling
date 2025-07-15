import torch
import torch.nn as nn
import time
import numpy as np
from src.resnet_transfer import ResNet50_Transfer
from src.efficientnet_transfer import EfficientNet_Transfer
from src.utils import get_cifar10_loaders, count_parameters
from config import *

def benchmark_model(model, data_loader, device, num_batches=50):
    """Benchmark model inference speed and accuracy"""
    model.eval()
    model.to(device)
    
    inference_times = []
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i, (data, target) in enumerate(data_loader):
            if i >= num_batches:
                break
                
            data, target = data.to(device), target.to(device)
            
            # Time inference
            start_time = time.time()
            output = model(data)
            inference_time = time.time() - start_time
            
            inference_times.append(inference_time)
            
            # Calculate accuracy
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    avg_inference_time = np.mean(inference_times) * 1000  # Convert to ms
    std_inference_time = np.std(inference_times) * 1000
    
    return {
        'accuracy': accuracy,
        'avg_inference_time_ms': avg_inference_time,
        'std_inference_time_ms': std_inference_time,
        'total_samples': total
    }

def main():
    device = torch.device(DEVICE)
    print(f"🔬 Comparing ResNet-50 vs EfficientNet-B3 on {device}")
    
    # Load test data
    _, test_loader = get_cifar10_loaders(DATA_DIR, BATCH_SIZE)
    
    # Initialize models
    resnet_model = ResNet50_Transfer(num_classes=10).to(device)
    efficientnet_model = EfficientNet_Transfer(num_classes=10, variant='b3').to(device)
    
    # Load trained weights if available
    try:
        resnet_model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/best_resnet50.pth"))
        print("✅ Loaded trained ResNet-50 weights")
    except:
        print("⚠️  Using untrained ResNet-50 weights")
    
    try:
        efficientnet_model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/best_efficientnet_b3.pth"))
        print("✅ Loaded trained EfficientNet-B3 weights")
    except:
        print("⚠️  Using untrained EfficientNet-B3 weights")
    
    # Model statistics
    resnet_params = count_parameters(resnet_model)
    efficientnet_params = count_parameters(efficientnet_model)
    
    print(f"\n📊 MODEL COMPARISON RESULTS")
    print("="*60)
    
    # Parameter comparison
    print(f"\n🔢 PARAMETER COUNT:")
    print(f"   ResNet-50:      {resnet_params[0]:,} total, {resnet_params[1]:,} trainable")
    print(f"   EfficientNet-B3: {efficientnet_params[0]:,} total, {efficientnet_params[1]:,} trainable")
    print(f"   Difference:     {abs(resnet_params[0] - efficientnet_params[0]):,} parameters")
    
    # Benchmark both models
    print(f"\n⚡ SPEED & ACCURACY BENCHMARK:")
    
    resnet_results = benchmark_model(resnet_model, test_loader, device)
    print(f"\n   ResNet-50:")
    print(f"     Accuracy: {resnet_results['accuracy']:.2f}%")
    print(f"     Inference: {resnet_results['avg_inference_time_ms']:.2f} ± {resnet_results['std_inference_time_ms']:.2f} ms/batch")
    
    efficientnet_results = benchmark_model(efficientnet_model, test_loader, device)
    print(f"\n   EfficientNet-B3:")
    print(f"     Accuracy: {efficientnet_results['accuracy']:.2f}%")
    print(f"     Inference: {efficientnet_results['avg_inference_time_ms']:.2f} ± {efficientnet_results['std_inference_time_ms']:.2f} ms/batch")
    
    # Summary
    print(f"\n🏆 WINNER ANALYSIS:")
    
    if resnet_results['accuracy'] > efficientnet_results['accuracy']:
        acc_winner = "ResNet-50"
        acc_diff = resnet_results['accuracy'] - efficientnet_results['accuracy']
    else:
        acc_winner = "EfficientNet-B3"
        acc_diff = efficientnet_results['accuracy'] - resnet_results['accuracy']
    
    if resnet_results['avg_inference_time_ms'] < efficientnet_results['avg_inference_time_ms']:
        speed_winner = "ResNet-50"
        speed_diff = efficientnet_results['avg_inference_time_ms'] - resnet_results['avg_inference_time_ms']
    else:
        speed_winner = "EfficientNet-B3"
        speed_diff = resnet_results['avg_inference_time_ms'] - efficientnet_results['avg_inference_time_ms']
    
    print(f"   Accuracy Winner: {acc_winner} (+{acc_diff:.2f}%)")
    print(f"   Speed Winner: {speed_winner} (+{speed_diff:.2f}ms faster)")
    
    # MANG-style recommendation
    print(f"\n💼 MANG PRODUCTION RECOMMENDATION:")
    if acc_diff < 2.0 and speed_diff > 5.0:  # If accuracy is close but speed differs significantly
        print(f"   Choose {speed_winner} - Speed matters more at scale when accuracy is comparable")
    elif acc_diff > 5.0:  # If accuracy difference is significant
        print(f"   Choose {acc_winner} - Accuracy improvement justifies speed trade-off")
    else:
        print(f"   Both models are comparable - choose based on team expertise and infrastructure")

if __name__ == "__main__":
    main()

