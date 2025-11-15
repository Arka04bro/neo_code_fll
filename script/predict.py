"""
YOLOv8 Prediction Script for Archaeological Object Detection
Скрипт для применения обученной модели к новым изображениям
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2


def predict_on_images(
    model_path: str,
    source: str,
    conf: float = 0.5,
    iou: float = 0.6,
    imgsz: int = 640,
    device: str = '0',
    save: bool = True,
    save_txt: bool = True,
    save_conf: bool = True,
    save_crop: bool = True,
    show: bool = False,
    project: str = 'predictions',
    name: str = 'exp'
):
    """
    Применение модели к изображениям/видео
    
    Args:
        model_path: Путь к обученной модели
        source: Путь к изображению/папке/видео
        conf: Порог уверенности
        iou: IoU порог для NMS
        imgsz: Размер изображений
        device: GPU device или 'cpu'
        save: Сохранить результаты
        save_txt: Сохранить аннотации в txt
        save_conf: Сохранить уверенность в txt
        save_crop: Сохранить кропы объектов
        show: Показать результаты
        project: Директория для результатов
        name: Название эксперимента
    """
    
    print("\n" + "="*80)
    print("ПРИМЕНЕНИЕ МОДЕЛИ К ИЗОБРАЖЕНИЯМ")
    print("="*80 + "\n")
    
    # Проверка модели
    if not Path(model_path).exists():
        print(f"❌ ОШИБКА: Модель не найдена: {model_path}")
        return None
    
    # Проверка источника
    source_path = Path(source)
    if not source_path.exists():
        print(f"❌ ОШИБКА: Источник не найден: {source}")
        return None
    
    print(f"[1/2] Загрузка модели: {model_path}")
    model = YOLO(model_path)
    
    print(f"[2/2] Обработка источника: {source}")
    
    # Подсчет файлов
    if source_path.is_file():
        print(f"  Тип: {'Видео' if source_path.suffix in ['.mp4', '.avi', '.mov'] else 'Изображение'}")
    elif source_path.is_dir():
        images = list(source_path.glob('*.jpg')) + list(source_path.glob('*.png'))
        print(f"  Тип: Директория")
        print(f"  Найдено изображений: {len(images)}")
    
    print("\nПараметры детекции:")
    print(f"  Порог уверенности: {conf}")
    print(f"  IoU порог: {iou}")
    print(f"  Размер изображений: {imgsz}")
    print("-"*80 + "\n")
    
    # Предсказание
    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        save=save,
        save_txt=save_txt,
        save_conf=save_conf,
        save_crop=save_crop,
        show=show,
        project=project,
        name=name,
        verbose=True
    )
    
    # Статистика
    total_detections = 0
    class_counts = {}
    
    for result in results:
        if result.boxes is not None:
            total_detections += len(result.boxes)
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
    
    print("\n" + "="*80)
    print("РЕЗУЛЬТАТЫ ДЕТЕКЦИИ")
    print("="*80 + "\n")
    print(f"Всего детекций: {total_detections}")
    
    if class_counts:
        print("\nРаспределение по классам:")
        for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls_name}: {count}")
    
    output_dir = Path(project) / name
    print(f"\nРезультаты сохранены в: {output_dir}")
    
    if save:
        print("  • Изображения с детекциями")
    if save_txt:
        print("  • Текстовые аннотации (YOLO format)")
    if save_crop:
        print("  • Кропы обнаруженных объектов")
    
    print("="*80 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Apply trained YOLOv8 model to new images'
    )
    
    parser.add_argument('--model', type=str, required=True,
                        help='Путь к обученной модели (.pt файл)')
    parser.add_argument('--source', type=str, required=True,
                        help='Путь к изображению/папке/видео')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Порог уверенности')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='IoU порог для NMS')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Размер изображений')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU device или cpu')
    parser.add_argument('--save', action='store_true', default=True,
                        help='Сохранить результаты')
    parser.add_argument('--save-txt', action='store_true', default=True,
                        help='Сохранить аннотации')
    parser.add_argument('--save-crop', action='store_true', default=True,
                        help='Сохранить кропы')
    parser.add_argument('--show', action='store_true',
                        help='Показать результаты')
    parser.add_argument('--project', type=str, default='predictions',
                        help='Директория для результатов')
    parser.add_argument('--name', type=str, default='exp',
                        help='Название эксперимента')
    
    args = parser.parse_args()
    
    predict_on_images(
        model_path=args.model,
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=True,
        save_crop=args.save_crop,
        show=args.show,
        project=args.project,
        name=args.name
    )


if __name__ == '__main__':
    main()
