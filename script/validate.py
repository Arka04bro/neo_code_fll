"""
YOLOv8 Validation Script for Archaeological Object Detection
Скрипт для валидации обученной модели на тестовом датасете
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import yaml
import json


def validate_model(
    model_path: str,
    data_config: str,
    split: str = 'val',
    imgsz: int = 640,
    batch: int = 16,
    conf: float = 0.001,
    iou: float = 0.6,
    device: str = '0',
    save_json: bool = True,
    save_hybrid: bool = False,
    plots: bool = True
):
    """
    Валидация модели на датасете
    
    Args:
        model_path: Путь к обученной модели
        data_config: Путь к конфигу датасета
        split: Какой split использовать (val/test)
        imgsz: Размер изображений
        batch: Размер батча
        conf: Порог уверенности
        iou: IoU порог для NMS
        device: GPU device или 'cpu'
        save_json: Сохранить результаты в JSON
        save_hybrid: Сохранить гибридные метки
        plots: Создать графики
    """
    
    print("\n" + "="*80)
    print("ВАЛИДАЦИЯ МОДЕЛИ YOLOV8")
    print("="*80 + "\n")
    
    # Проверка существования модели
    if not Path(model_path).exists():
        print(f"❌ ОШИБКА: Модель не найдена: {model_path}")
        return None
    
    print(f"[1/3] Загрузка модели: {model_path}")
    model = YOLO(model_path)
    
    # Загрузка конфига датасета
    print(f"[2/3] Загрузка конфигурации датасета: {data_config}")
    with open(data_config, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"  Классы: {len(dataset_config['names'])}")
    for idx, name in dataset_config['names'].items():
        print(f"    {idx}: {name}")
    
    # Валидация
    print(f"\n[3/3] Запуск валидации на split: {split}")
    print("-"*80)
    
    results = model.val(
        data=data_config,
        split=split,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        save_json=save_json,
        save_hybrid=save_hybrid,
        plots=plots,
        verbose=True
    )
    
    # Вывод результатов
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
    print("="*80 + "\n")
    
    # Общие метрики
    print("Общие метрики:")
    print(f"  mAP@0.5:0.95: {results.box.map:.4f} ({results.box.map*100:.2f}%)")
    print(f"  mAP@0.5:     {results.box.map50:.4f} ({results.box.map50*100:.2f}%)")
    print(f"  mAP@0.75:    {results.box.map75:.4f} ({results.box.map75*100:.2f}%)")
    print(f"  Precision:   {results.box.mp:.4f} ({results.box.mp*100:.2f}%)")
    print(f"  Recall:      {results.box.mr:.4f} ({results.box.mr*100:.2f}%)")
    
    # Метрики по классам
    print("\nМетрики по классам:")
    print("-"*80)
    print(f"{'Класс':<20} {'AP@0.5':<10} {'AP@0.5:0.95':<15} {'Precision':<12} {'Recall':<10}")
    print("-"*80)
    
    for idx, class_name in dataset_config['names'].items():
        ap50 = results.box.ap50[idx]
        ap = results.box.ap[idx].mean()  # среднее по IoU порогам
        precision = results.box.p[idx]
        recall = results.box.r[idx]
        
        print(f"{class_name:<20} {ap50:<10.4f} {ap:<15.4f} {precision:<12.4f} {recall:<10.4f}")
    
    print("="*80 + "\n")
    
    # Сохранение результатов в JSON
    if save_json:
        results_dict = {
            'model': str(model_path),
            'dataset': str(data_config),
            'split': split,
            'metrics': {
                'mAP_50_95': float(results.box.map),
                'mAP_50': float(results.box.map50),
                'mAP_75': float(results.box.map75),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr)
            },
            'per_class': {}
        }
        
        for idx, class_name in dataset_config['names'].items():
            results_dict['per_class'][class_name] = {
                'ap50': float(results.box.ap50[idx]),
                'ap_50_95': float(results.box.ap[idx].mean()),
                'precision': float(results.box.p[idx]),
                'recall': float(results.box.r[idx])
            }
        
        output_path = Path(model_path).parent.parent / 'validation_results.json'
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"✓ Результаты сохранены в: {output_path}\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate YOLOv8 model for archaeological object detection'
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Путь к обученной модели (.pt файл)')
    parser.add_argument('--data', type=str,
                        default='configs/archaeological_dataset.yaml',
                        help='Путь к конфигу датасета')
    parser.add_argument('--split', type=str, default='val',
                        choices=['val', 'test'],
                        help='Какой split использовать')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Размер изображений')
    parser.add_argument('--batch', type=int, default=16,
                        help='Размер батча')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Порог уверенности')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='IoU порог для NMS')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device или cpu')
    parser.add_argument('--save-json', action='store_true', default=True,
                        help='Сохранить результаты в JSON')
    parser.add_argument('--plots', action='store_true', default=True,
                        help='Создать графики')
    
    args = parser.parse_args()
    
    validate_model(
        model_path=args.model,
        data_config=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_json=args.save_json,
        plots=args.plots
    )


if __name__ == '__main__':
    main()
