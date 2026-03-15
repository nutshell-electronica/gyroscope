# Comandos para giroscopio_pose_v2.py

## Lanzamiento normal
```bash
cd /home/ai
python3 giroscopio_pose_v2.py
```

## Lanzar en segundo plano (sin bloquear la terminal)
```bash
nohup python3 giroscopio_pose_v2.py > /tmp/giroscopio.log 2>&1 &
```

## Ver log en tiempo real (si se lanzó con nohup)
```bash
tail -f /tmp/giroscopio.log
```

## Detener si se lanzó en segundo plano
```bash
pkill -f giroscopio_pose_v2.py
```

## Lanzar con display explícito (si la ventana no abre)
```bash
DISPLAY=:0 python3 giroscopio_pose_v2.py
```

## Lanzar al arrancar la Raspberry (autostart)
```bash
# Añadir al crontab:
crontab -e
# Añadir esta línea al final:
@reboot sleep 10 && DISPLAY=:0 python3 /home/ai/giroscopio_pose_v2.py >> /tmp/giroscopio.log 2>&1
```

## Ver información del chip Hailo
```bash
hailortcli fw-control identify
```

## Comprobar que el Hailo está conectado
```bash
hailortcli scan
```

## Ver temperatura y uso del chip Hailo
```bash
hailortcli monitor
```

## Comprobar cámara disponible
```bash
libcamera-hello --list-cameras
```

## Probar la cámara (preview 5 segundos)
```bash
libcamera-hello -t 5000
```

## Ver versión de HailoRT
```bash
python3 -c "import hailo_platform; print(hailo_platform.__version__)"
```

## Copiar script desde el PC (desde Windows con WinSCP o desde Linux/Mac)
```bash
scp giroscopio_pose_v2.py ai@ai-pi.local:/home/ai/
```

## Hacer backup del script actual antes de modificarlo
```bash
cp /home/ai/giroscopio_pose_v2.py /home/ai/giroscopio_pose_v2_backup_$(date +%Y%m%d).py
```
