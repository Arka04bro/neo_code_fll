"""
Dataset Preparation Utilities for YOLOv8 Archaeological Detection
Утилиты для подготовки датасета археологических объектов
"""

import os
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Tuple
import json
import yaml
from PIL import Image
import cv2


def split_dataset(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Разделение датасета на train/val/test
    
    Args:
        images_dir: Директория с изображениями
        labels_dir: Директория с аннотациями (YOLO format)
        output_dir: Выходная директория
        train_ratio: Доля обучающей выборки
        val_ratio: Доля валидационной выборки
        test_ratio: Доля тестовой выборки
        seed: Seed для воспроизводимости
    """
    
    print("\n" + "="*80)
    print("РАЗДЕЛЕНИЕ ДАТАСЕТА НА TRAIN/VAL/TEST")
    print("="*80 + "\n")
    
    # Проверка соотношений
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.01, \
        "Сумма соотношений должна быть равна 1.0"
    
    # Получение списка изображений
    images_path = Path(images_dir)
    images = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
    
    print(f"Найдено изображений: {len(images)}")
    
    if len(images) == 0:
        print("❌ ОШИБКА: Нет изображений в директории")
        return
    
    # Проверка наличия аннотаций
    labels_path = Path(labels_dir)
    images_with_labels = []
    images_without_labels = []
    
    for img_path in images:
        label_path = labels_path / (img_path.stem + '.txt')
        if label_path.exists():
            images_with_labels.append(img_path)
        else:
            images_without_labels.append(img_path)
    
    print(f"Изображений с аннотациями: {len(images_with_labels)}")
    print(f"Изображений без аннотаций: {len(images_without_labels)}")
    
    if len(images_without_labels) > 0:
        print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: {len(images_without_labels)} изображений без аннотаций будут пропущены")
    
    # Перемешивание
    random.seed(seed)
    random.shuffle(images_with_labels)
    
    # Разделение
    n_total = len(images_with_labels)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    train_images = images_with_labels[:n_train]
    val_images = images_with_labels[n_train:n_train + n_val]
    test_images = images_with_labels[n_train + n_val:]
    
    print(f"\nРазделение:")
    print(f"  Train: {len(train_images)} ({len(train_images)/n_total*100:.1f}%)")
    print(f"  Val:   {len(val_images)} ({len(val_images)/n_total*100:.1f}%)")
    print(f"  Test:  {len(test_images)} ({len(test_images)/n_total*100:.1f}%)")
    
    # Создание структуры директорий
    output_path = Path(output_dir)
    for split in ['train', 'val', 'test']:
        (output_path / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output_path / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Копирование файлов
    print("\nКопирование файлов...")
    
    def copy_files(image_list: List[Path], split: str):
        for img_path in image_list:
            # Копирование изображения
            dst_img = output_path / 'images' / split / img_path.name
            shutil.copy2(img_path, dst_img)
            
            # Копирование аннотации
            label_path = labels_path / (img_path.stem + '.txt')
            dst_label = output_path / 'labels' / split / (img_path.stem + '.txt')
            shutil.copy2(label_path, dst_label)
    
    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    copy_files(test_images, 'test')
    
    print("✓ Датасет успешно разделен")
    print(f"✓ Результаты сохранены в: {output_dir}")
    print("="*80 + "\n")


def analyze_dataset(dataset_dir: str, config_path: str):
    """
    Анализ датасета и вывод статистики
    
    Args:
        dataset_dir: Корневая директория датасета
        config_path: Путь к конфигу датасета
    """
    
    print("\n" + "="*80)
    print("АНАЛИЗ ДАТАСЕТА")
    print("="*80 + "\n")
    
    # Загрузка конфига
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config['names']
    dataset_path = Path(dataset_dir)
    
    # Анализ каждого split
    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split
        
        if not images_dir.exists():
            continue
        
        print(f"\n{split.upper()}:")
        print("-"*80)
        
        # Подсчет изображений
        images = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
        print(f"Изображений: {len(images)}")
        
        # Анализ аннотаций
        class_counts = {cls_id: 0 for cls_id in class_names.keys()}
        total_objects = 0
        image_sizes = []
        objects_per_image = []
        
        for img_path in images:
            # Размер изображения
            img = Image.open(img_path)
            image_sizes.append(img.size)
            
            # Аннотации
            label_path = labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    objects_per_image.append(len(lines))
                    total_objects += len(lines)
                    
                    for line in lines:
                        cls_id = int(line.split()[0])
                        if cls_id in class_counts:
                            class_counts[cls_id] += 1
        
        print(f"Всего объектов: {total_objects}")
        print(f"Объектов на изображение (среднее): {total_objects/len(images):.2f}")
        
        # Распределение по классам
        print("\nРаспределение по классам:")
        for cls_id, count in class_counts.items():
            cls_name = class_names[cls_id]
            percentage = (count / total_objects * 100) if total_objects > 0 else 0
            print(f"  {cls_name:<20} {count:>6} ({percentage:>5.1f}%)")
        
        # Размеры изображений
        if image_sizes:
            widths = [size[0] for size in image_sizes]
            heights = [size[1] for size in image_sizes]
            print(f"\nРазмеры изображений:")
            print(f"  Ширина:  {min(widths)} - {max(widths)} (среднее: {sum(widths)/len(widths):.0f})")
            print(f"  Высота:  {min(heights)} - {max(heights)} (среднее: {sum(heights)/len(heights):.0f})")
    
    print("\n" + "="*80 + "\n")


def convert_coco_to_yolo(
    coco_json: str,
    images_dir: str,
    output_dir: str
):
    """
    Конвертация аннотаций из формата COCO в формат YOLO
    
    Args:
        coco_json: Путь к COCO JSON файлу
        images_dir: Директория с изображениями
        output_dir: Выходная директория
    """
    
    print("\n" + "="*80)
    print("КОНВЕРТАЦИЯ COCO → YOLO")
    print("="*80 + "\n")
    
    # Загрузка COCO JSON
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    
    # Создание маппинга категорий
    category_mapping = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    print(f"Категорий: {len(coco_data['categories'])}")
    print(f"Изображений: {len(coco_data['images'])}")
    print(f"Аннотаций: {len(coco_data['annotations'])}")
    
    # Создание выходной директории
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Группировка аннотаций по изображениям
    annotations_by_image = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)
    
    # Конвертация
    print("\nКонвертация аннотаций...")
    converted = 0
    
    for img_info in coco_data['images']:
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Получение аннотаций для этого изображения
        if img_id not in annotations_by_image:
            continue
        
        # Создание YOLO аннотации
        yolo_lines = []
        for ann in annotations_by_image[img_id]:
            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann['bbox']
            
            # Конвертация в YOLO format: [class_id, x_center, y_center, width, height] (нормализованные)
            x_center = (x + w / 2) / img_width
            y_center = (y + h / 2) / img_height
            norm_width = w / img_width
            norm_height = h / img_height
            
            class_id = category_mapping[ann['category_id']]
            
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")
        
        # Сохранение аннотации
        label_filename = Path(img_filename).stem + '.txt'
        label_path = output_path / label_filename
        
        with open(label_path, 'w') as f:
            f.writelines(yolo_lines)
        
        converted += 1
    
    print(f"✓ Сконвертировано аннотаций: {converted}")
    print(f"✓ Результаты сохранены в: {output_dir}")
    print("="*80 + "\n")


def visualize_annotations(
    images_dir: str,
    labels_dir: str,
    output_dir: str,
    config_path: str,
    num_samples: int = 10
):
    """
    Визуализация аннотаций на изображениях
    
    Args:
        images_dir: Директория с изображениями
        labels_dir: Директория с аннотациями
        output_dir: Директория для сохранения визуализаций
        config_path: Путь к конфигу датасета
        num_samples: Количество примеров для визуализации
    """
    
    print("\n" + "="*80)
    print("ВИЗУАЛИЗАЦИЯ АННОТАЦИЙ")
    print("="*80 + "\n")
    
    # Загрузка конфига
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    class_names = config['names']
    
    # Получение списка изображений
    images_path = Path(images_dir)
    images = list(images_path.glob('*.jpg')) + list(images_path.glob('*.png'))
    
    # Случайная выборка
    random.shuffle(images)
    images = images[:num_samples]
    
    print(f"Визуализация {len(images)} примеров...")
    
    # Создание выходной директории
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    labels_path = Path(labels_dir)
    
    for img_path in images:
        # Загрузка изображения
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]
        
        # Загрузка аннотаций
        label_path = labels_path / (img_path.stem + '.txt')
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    cls_id = int(parts[0])
                    x_center, y_center, bbox_width, bbox_height = map(float, parts[1:5])
                    
                    # Денормализация координат
                    x_center *= width
                    y_center *= height
                    bbox_width *= width
                    bbox_height *= height
                    
                    # Вычисление углов рамки
                    x1 = int(x_center - bbox_width / 2)
                    y1 = int(y_center - bbox_height / 2)
                    x2 = int(x_center + bbox_width / 2)
                    y2 = int(y_center + bbox_height / 2)
                    
                    # Рисование рамки
                    color = (0, 255, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Добавление метки класса
                    cls_name = class_names[cls_id]
                    label = f"{cls_name}"
                    
                    # Фон для текста
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(img, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Сохранение
        output_file = output_path / img_path.name
        cv2.imwrite(str(output_file), img)
    
    print(f"✓ Визуализации сохранены в: {output_dir}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Dataset preparation utilities for YOLOv8'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Команда')
    
    # Split dataset
    split_parser = subparsers.add_parser('split', help='Разделить датасет на train/val/test')
    split_parser.add_argument('--images', required=True, help='Директория с изображениями')
    split_parser.add_argument('--labels', required=True, help='Директория с аннотациями')
    split_parser.add_argument('--output', required=True, help='Выходная директория')
    split_parser.add_argument('--train-ratio', type=float, default=0.7, help='Доля train')
    split_parser.add_argument('--val-ratio', type=float, default=0.2, help='Доля val')
    split_parser.add_argument('--test-ratio', type=float, default=0.1, help='Доля test')
    split_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Analyze dataset
    analyze_parser = subparsers.add_parser('analyze', help='Анализировать датасет')
    analyze_parser.add_argument('--dataset', required=True, help='Директория датасета')
    analyze_parser.add_argument('--config', required=True, help='Конфиг датасета')
    
    # Convert COCO to YOLO
    convert_parser = subparsers.add_parser('convert', help='Конвертировать COCO в YOLO')
    convert_parser.add_argument('--coco-json', required=True, help='COCO JSON файл')
    convert_parser.add_argument('--images', required=True, help='Директория с изображениями')
    convert_parser.add_argument('--output', required=True, help='Выходная директория')
    
    # Visualize annotations
    viz_parser = subparsers.add_parser('visualize', help='Визуализировать аннотации')
    viz_parser.add_argument('--images', required=True, help='Директория с изображениями')
    viz_parser.add_argument('--labels', required=True, help='Директория с аннотациями')
    viz_parser.add_argument('--output', required=True, help='Директория для визуализаций')
    viz_parser.add_argument('--config', required=True, help='Конфиг датасета')
    viz_parser.add_argument('--num-samples', type=int, default=10, help='Количество примеров')
    
    args = parser.parse_args()
    
    if args.command == 'split':
        split_dataset(
            args.images, args.labels, args.output,
            args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
    elif args.command == 'analyze':
        analyze_dataset(args.dataset, args.config)
    elif args.command == 'convert':
        convert_coco_to_yolo(args.coco_json, args.images, args.output)
    elif args.command == 'visualize':
        visualize_annotations(
            args.images, args.labels, args.output, args.config, args.num_samples
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
