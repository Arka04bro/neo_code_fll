"""
YOLOv8 Fine-tuning Script for Archaeological Object Detection
Скрипт для дообучения модели YOLOv8 на археологических объектах
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import torch


def load_config(config_path: str) -> dict:
    """Загрузка конфигурации из YAML файла"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def check_dataset(data_config: str) -> bool:
    """Проверка наличия датасета"""
    with open(data_config, 'r', encoding='utf-8') as f:
        dataset = yaml.safe_load(f)
    
    dataset_path = Path(dataset['path'])
    train_path = dataset_path / dataset['train']
    val_path = dataset_path / dataset['val']
    
    if not train_path.exists():
        print(f"❌ ОШИБКА: Директория с обучающими изображениями не найдена: {train_path}")
        return False
    
    if not val_path.exists():
        print(f"❌ ОШИБКА: Директория с валидационными изображениями не найдена: {val_path}")
        return False
    
    # Подсчет изображений
    train_images = list(train_path.glob('*.jpg')) + list(train_path.glob('*.png'))
    val_images = list(val_path.glob('*.jpg')) + list(val_path.glob('*.png'))
    
    print(f"✓ Найдено обучающих изображений: {len(train_images)}")
    print(f"✓ Найдено валидационных изображений: {len(val_images)}")
    
    if len(train_images) == 0:
        print("❌ ОШИБКА: Нет обучающих изображений!")
        return False
    
    if len(val_images) == 0:
        print("⚠️  ПРЕДУПРЕЖДЕНИЕ: Нет валидационных изображений!")
    
    return True


def setup_training_environment():
    """Настройка окружения для обучения"""
    # Проверка доступности GPU
    if torch.cuda.is_available():
        print(f"✓ GPU доступен: {torch.cuda.get_device_name(0)}")
        print(f"  Память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️  GPU не доступен. Обучение будет на CPU (медленно)")
    
    # Проверка версии ultralytics
    try:
        import ultralytics
        print(f"✓ Ultralytics YOLOv8 версия: {ultralytics.__version__}")
    except ImportError:
        print("❌ ОШИБКА: Ultralytics не установлен. Установите: pip install ultralytics")
        sys.exit(1)


def train_model(
    model_name: str = 'yolov8n.pt',
    data_config: str = 'configs/archaeological_dataset.yaml',
    epochs: int = 100,
    batch: int = 16,
    imgsz: int = 640,
    device: str = '0',
    project: str = 'results',
    name: str = 'archaeological_detection',
    resume: bool = False,
    **kwargs
):
    """
    Обучение модели YOLOv8
    
    Args:
        model_name: Название предобученной модели
        data_config: Путь к конфигу датасета
        epochs: Количество эпох
        batch: Размер батча
        imgsz: Размер изображений
        device: GPU device или 'cpu'
        project: Директория для результатов
        name: Название эксперимента
        resume: Продолжить обучение
        **kwargs: Дополнительные параметры
    """
    
    print("\n" + "="*80)
    print("ОБУЧЕНИЕ МОДЕЛИ YOLOV8 ДЛЯ ДЕТЕКЦИИ АРХЕОЛОГИЧЕСКИХ ОБЪЕКТОВ")
    print("="*80 + "\n")
    
    # Загрузка модели
    print(f"[1/4] Загрузка модели: {model_name}")
    
    if resume and Path(project) / name / 'weights' / 'last.pt':
        # Продолжение обучения
        model_path = Path(project) / name / 'weights' / 'last.pt'
        print(f"  Продолжение обучения с чекпоинта: {model_path}")
        model = YOLO(str(model_path))
    else:
        # Новое обучение
        model = YOLO(model_name)
        print(f"  Загружена предобученная модель: {model_name}")
    
    # Проверка датасета
    print(f"\n[2/4] Проверка датасета: {data_config}")
    if not check_dataset(data_config):
        print("\n❌ Обучение прервано из-за проблем с датасетом")
        sys.exit(1)
    
    # Настройка окружения
    print(f"\n[3/4] Настройка окружения")
    setup_training_environment()
    
    # Запуск обучения
    print(f"\n[4/4] Запуск обучения")
    print("-"*80)
    print(f"Параметры:")
    print(f"  Эпохи: {epochs}")
    print(f"  Размер батча: {batch}")
    print(f"  Размер изображений: {imgsz}x{imgsz}")
    print(f"  Устройство: {device}")
    print(f"  Результаты: {project}/{name}")
    print("-"*80 + "\n")
    
    # Время начала
    start_time = datetime.now()
    
    try:
        # Обучение
        results = model.train(
            data=data_config,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            verbose=True,
            **kwargs
        )
        
        # Время окончания
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("="*80)
        print(f"Время обучения: {duration}")
        print(f"Результаты сохранены в: {project}/{name}")
        print("\nСозданные файлы:")
        print(f"  • {project}/{name}/weights/best.pt - лучшая модель")
        print(f"  • {project}/{name}/weights/last.pt - последняя модель")
        print(f"  • {project}/{name}/results.png - графики обучения")
        print(f"  • {project}/{name}/confusion_matrix.png - матрица ошибок")
        print("="*80 + "\n")
        
        return results
        
    except Exception as e:
        print(f"\n❌ ОШИБКА ВО ВРЕМЯ ОБУЧЕНИЯ: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLOv8 for archaeological object detection'
    )
    
    # Основные параметры
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help='Предобученная модель (yolov8n/s/m/l/x.pt)')
    parser.add_argument('--data', type=str, 
                        default='configs/archaeological_dataset.yaml',
                        help='Путь к конфигу датасета')
    parser.add_argument('--config', type=str, 
                        default='configs/training_config.yaml',
                        help='Путь к конфигу обучения')
    
    # Параметры обучения
    parser.add_argument('--epochs', type=int, default=100,
                        help='Количество эпох обучения')
    parser.add_argument('--batch', type=int, default=16,
                        help='Размер батча')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Размер входных изображений')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device (0, 1, ...) или cpu')
    
    # Пути
    parser.add_argument('--project', type=str, default='results',
                        help='Директория для результатов')
    parser.add_argument('--name', type=str, default='archaeological_detection',
                        help='Название эксперимента')
    
    # Дополнительные опции
    parser.add_argument('--resume', action='store_true',
                        help='Продолжить обучение с последнего чекпоинта')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Использовать предобученные веса')
    
    args = parser.parse_args()
    
    # Загрузка конфига если указан
    if args.config and Path(args.config).exists():
        print(f"Загрузка конфигурации из: {args.config}")
        config = load_config(args.config)
        
        # Обновление параметров из конфига
        for key, value in config.items():
            if key not in ['archaeological_settings', 'dataset_info'] and hasattr(args, key):
                setattr(args, key, value)
    
    # Запуск обучения
    train_model(
        model_name=args.model,
        data_config=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
        pretrained=args.pretrained
    )


if __name__ == '__main__':
    main()
