#!/usr/bin/env python3
"""
FPV Drone Archaeological Detection System
Главный скрипт для запуска системы компьютерного зрения
"""

import argparse
import sys
from pathlib import Path

# Добавление путей к модулям
sys.path.insert(0, str(Path(__file__).parent))

from models.archaeological_detector import ArchaeologicalDetector
from utils.video_processor import VideoStreamProcessor, RTSPStreamProcessor
from utils.drone_integration import ArchaeologyMissionController


def main():
    """Главная функция приложения"""
    parser = argparse.ArgumentParser(
        description="FPV Drone Archaeological Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:

  # Обработка с веб-камеры
  python main.py --source 0 --display

  # Обработка RTSP-потока с дрона
  python main.py --source rtsp://192.168.1.100:8554/stream --rtsp

  # Обработка видеофайла с сохранением
  python main.py --source video.mp4 --output result.mp4

  # С подключением к дрону через MAVLink
  python main.py --source 0 --drone-connection /dev/ttyUSB0 --no-simulation

  # Только детекция на изображении
  python main.py --image test.jpg --output detected.jpg

Управление во время работы:
  Q - выход
  P - пауза/возобновление
  S - сохранить снимок
        """
    )
    
    # Источник видео
    parser.add_argument(
        '--source',
        type=str,
        default='0',
        help='Источник видео: номер камеры (0, 1, ...), путь к файлу или RTSP URL'
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='Путь к изображению для детекции (вместо видео)'
    )
    
    # Параметры модели
    parser.add_argument(
        '--model',
        type=str,
        default='yolov8n.pt',
        help='Путь к весам модели YOLO (по умолчанию: yolov8n.pt)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Порог уверенности для детекции (0-1, по умолчанию: 0.5)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Устройство для вычислений (по умолчанию: cpu)'
    )
    
    # Параметры видео
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Целевой FPS для обработки (по умолчанию: 30)'
    )
    
    parser.add_argument(
        '--resolution',
        type=str,
        default='1280x720',
        help='Разрешение видео в формате WIDTHxHEIGHT (по умолчанию: 1280x720)'
    )
    
    parser.add_argument(
        '--rtsp',
        action='store_true',
        help='Использовать RTSP-процессор с переподключением'
    )
    
    # Вывод
    parser.add_argument(
        '--output',
        type=str,
        help='Путь для сохранения обработанного видео'
    )
    
    parser.add_argument(
        '--display',
        action='store_true',
        help='Отображать видео в окне'
    )
    
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Не отображать видео (для headless режима)'
    )
    
    # Сохранение детекций
    parser.add_argument(
        '--save-detections',
        action='store_true',
        default=True,
        help='Сохранять вырезки обнаруженных объектов'
    )
    
    parser.add_argument(
        '--detections-dir',
        type=str,
        default='./detections',
        help='Директория для сохранения детекций'
    )
    
    # Интеграция с дроном
    parser.add_argument(
        '--drone-connection',
        type=str,
        help='Строка подключения к дрону (например: /dev/ttyUSB0, 127.0.0.1:14550)'
    )
    
    parser.add_argument(
        '--no-simulation',
        action='store_true',
        help='Отключить режим симуляции GPS (использовать реальное подключение)'
    )
    
    parser.add_argument(
        '--mission-name',
        type=str,
        help='Название миссии для логов'
    )
    
    parser.add_argument(
        '--log-dir',
        type=str,
        default='./logs',
        help='Директория для логов GPS и детекций'
    )
    
    args = parser.parse_args()
    
    # Парсинг разрешения
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except:
        print(f"Ошибка: неверный формат разрешения '{args.resolution}'")
        print("Используйте формат WIDTHxHEIGHT, например: 1920x1080")
        return 1
    
    print("="*60)
    print("FPV Drone Archaeological Detection System")
    print("="*60)
    
    # Инициализация детектора
    print("\n[1/3] Инициализация детектора...")
    try:
        detector = ArchaeologicalDetector(
            model_path=args.model,
            confidence_threshold=args.confidence,
            device=args.device
        )
        print(f"✓ Детектор загружен: {args.model}")
    except Exception as e:
        print(f"✗ Ошибка загрузки детектора: {e}")
        return 1
    
    # Инициализация контроллера миссии
    mission_controller = None
    if args.drone_connection or not args.no_simulation:
        print("\n[2/3] Инициализация контроллера миссии...")
        try:
            mission_controller = ArchaeologyMissionController(
                mission_name=args.mission_name,
                log_dir=args.log_dir,
                drone_connection=args.drone_connection,
                simulation_mode=not args.no_simulation
            )
            mission_controller.start_mission()
            print("✓ Контроллер миссии запущен")
        except Exception as e:
            print(f"✗ Ошибка инициализации миссии: {e}")
            print("Продолжение без GPS-логирования")
    
    # Обработка одиночного изображения
    if args.image:
        print(f"\n[3/3] Обработка изображения: {args.image}")
        try:
            import cv2
            
            # Загрузка изображения
            frame = cv2.imread(args.image)
            if frame is None:
                print(f"✗ Не удалось загрузить изображение: {args.image}")
                return 1
            
            # Детекция
            annotated_frame, detections = detector.detect(
                frame,
                draw_boxes=True,
                save_crops=args.save_detections,
                output_dir=args.detections_dir
            )
            
            print(f"✓ Обнаружено объектов: {len(detections)}")
            
            # Вывод детекций
            for det in detections:
                print(f"  - {det['class_name']}: {det['confidence']:.2f}")
            
            # Сохранение результата
            if args.output:
                cv2.imwrite(args.output, annotated_frame)
                print(f"✓ Результат сохранён: {args.output}")
            
            # Отображение
            if args.display:
                cv2.imshow('Detection Result', annotated_frame)
                print("\nНажмите любую клавишу для выхода...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            # Логирование детекций
            if mission_controller:
                for det in detections:
                    mission_controller.log_detection(det)
            
        except Exception as e:
            print(f"✗ Ошибка обработки изображения: {e}")
            return 1
        
        finally:
            if mission_controller:
                mission_controller.stop_mission()
        
        return 0
    
    # Обработка видеопотока
    print(f"\n[3/3] Инициализация видеопроцессора...")
    
    # Выбор процессора
    if args.rtsp and args.source.startswith('rtsp://'):
        processor = RTSPStreamProcessor(
            rtsp_url=args.source,
            target_fps=args.fps,
            resolution=resolution
        )
        print(f"✓ RTSP-процессор инициализирован")
    else:
        processor = VideoStreamProcessor(
            source=args.source,
            target_fps=args.fps,
            resolution=resolution
        )
        print(f"✓ Видеопроцессор инициализирован")
    
    # Callback для обработки кадров
    def process_frame(frame):
        """Обработка кадра с детекцией и логированием"""
        annotated_frame, detections = detector.detect(
            frame,
            draw_boxes=True,
            save_crops=args.save_detections,
            output_dir=args.detections_dir
        )
        
        # Логирование детекций
        if mission_controller and detections:
            for det in detections:
                mission_controller.log_detection(det)
        
        return annotated_frame, detections
    
    processor.set_process_callback(process_frame)
    
    print("\n" + "="*60)
    print("Система запущена!")
    print("="*60)
    print(f"Источник: {args.source}")
    print(f"Модель: {args.model}")
    print(f"Устройство: {args.device}")
    print(f"Разрешение: {resolution[0]}x{resolution[1]}")
    print(f"FPS: {args.fps}")
    
    if mission_controller:
        print(f"GPS-логирование: {'Включено' if not args.no_simulation else 'Реальное подключение'}")
    
    print("\nУправление:")
    print("  Q - выход")
    print("  P - пауза/возобновление")
    print("  S - сохранить снимок")
    print("="*60 + "\n")
    
    try:
        # Запуск обработки
        processor.start()
        
        # Отображение (если требуется)
        if args.display or not args.no_display:
            processor.display_loop()
        else:
            # Headless режим
            print("Headless режим. Нажмите Ctrl+C для остановки.")
            import time
            while processor.is_running:
                time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\nПолучен сигнал прерывания...")
    
    except Exception as e:
        print(f"\n✗ Ошибка во время работы: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Остановка процессора
        print("\nОстановка системы...")
        processor.stop()
        
        # Остановка миссии
        if mission_controller:
            mission_controller.stop_mission()
        
        # Сохранение лога детекций
        log_file = Path(args.log_dir) / "detections_log.json"
        detector.save_detections_log(str(log_file))
        
        # Статистика
        print("\n" + "="*60)
        print("Статистика работы")
        print("="*60)
        
        proc_stats = processor.get_statistics()
        for key, value in proc_stats.items():
            print(f"{key}: {value}")
        
        det_stats = detector.get_statistics()
        print(f"\nВсего детекций: {det_stats['total_detections']}")
        print(f"Средняя уверенность: {det_stats['average_confidence']:.2f}")
        
        if det_stats['class_distribution']:
            print("\nРаспределение по классам:")
            for class_name, count in det_stats['class_distribution'].items():
                print(f"  {class_name}: {count}")
        
        print("="*60)
        print("Система остановлена")
        print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
