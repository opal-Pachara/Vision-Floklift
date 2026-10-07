# 🐳 YOLOv12n AI Safety Monitoring — คู่มือการ Deploy ด้วย Docker

ระบบได้ถูกแพ็กเป็น Docker Container ที่ **ฝังโมเดล YOLOv12n (`best.onnx`) ไว้ภายในอิมเมจเรียบร้อยแล้ว** พร้อมทั้งคอมไพล์ **C++ Inference Engine (`libyolo_detector.so`)** และรัน Web Dashboard (FastAPI + OpenCV) ได้ทันทีโดยไม่ต้องติดตั้ง Dependency ภายนอกเพิ่มเติม

---

## 📁 ไฟล์ที่เกี่ยวข้องกับการ Deploy

| ไฟล์ | หน้าที่ |
| :--- | :--- |
| [Dockerfile](file:///Users/phatchara/Desktop/MasterDegree_Logistic/vision-logistic/Dockerfile) | สร้าง Ubuntu 22.04 + ONNX Runtime C++ + OpenCV + โมเดลฝังในตัว |
| [docker-compose.yml](file:///Users/phatchara/Desktop/MasterDegree_Logistic/vision-logistic/docker-compose.yml) | คอนฟิกสำหรับ Deploy 1 คำสั่งด้วย Docker Compose |
| [docker-build.sh](file:///Users/phatchara/Desktop/MasterDegree_Logistic/vision-logistic/docker-build.sh) | สคริปต์ช่วย Build Docker Image (`yolo-safety-monitor:latest`) |
| [docker-run.sh](file:///Users/phatchara/Desktop/MasterDegree_Logistic/vision-logistic/docker-run.sh) | สคริปต์รัน Container พร้อมตั้งค่า Port และ Volume Mount |
| [.dockerignore](file:///Users/phatchara/Desktop/MasterDegree_Logistic/vision-logistic/.dockerignore) | ป้องกันการคัดลอกไฟล์ขยะและไฟล์ binary ของ macOS เข้าไปใน Linux |

---

## 🚀 วิธีการรัน (2 วิธี)

### วิธีที่ 1: ใช้ Docker Compose (แนะนำ สะดวกที่สุด)

```bash
# สั่ง Build และรัน Background ทันที
docker compose up -d --build

# ดู Log การทำงาน
docker compose logs -f

# หยุดการทำงาน
docker compose down
```

เปิด Browser ไปที่:
👉 **`http://localhost:8080`**

---

### วิธีที่ 2: ใช้ Helper Scripts

```bash
# 1. Build Image (ฝังโมเดล best.onnx และคอมไพล์ C++ Engine)
./docker-build.sh

# 2. รัน Container (ค่าเริ่มต้น Port 8080)
./docker-run.sh

# (หรือระบุ Port ที่ต้องการ เช่น 9000)
./docker-run.sh 9000
```

---

## 📦 โครงสร้างสิ่งที่ฝังอยู่ใน Docker Image

1. **Embedded Model & Labels**:
   * `/app/best.onnx` (YOLOv12n ONNX Opset 18)
   * `/app/best.onnx.data` (Tensor weights)
   * `/app/labels.txt` (`forklift`, `helmet`, `person`, `safety_vest`)
2. **C++ Native Engine**:
   * ติดตั้ง ONNX Runtime C++ Library v1.18.0 อัตโนมัติ (รองรับทั้ง Intel/AMD `x86_64` และ ARM `aarch64`)
   * คอมไพล์ซอร์สโค้ด C++ เป็น `/app/build/libyolo_detector.so`
3. **Web & Backend**:
   * FastAPI Server พร้อมเชื่อมต่อ C++ ผ่าน C API (`yolo_cpp.py`)
   * หน้าเว็บ Modern Industrial Dark Theme (`/app/web`)
4. **Data Persistence**:
   * คอนฟิกใน `./config/settings.json` และผลลัพธ์ใน `./results/` จะถูก Sync กับ Host เครื่องหลัก ทำให้การปรับ Threshold หรือดาวน์โหลดไฟล์ทดสอบไม่สูญหายเมื่อรีสตาร์ตคอนเทนเนอร์

---

## 🔍 ตรวจสอบสถานะการทำงาน (Healthcheck & API)

```bash
# ตรวจสอบสถานะ Container
docker ps

# ทดสอบ API สถานะระบบ
curl http://localhost:8080/api/status

# ทดสอบทดลอง Detect รูปภาพผ่าน curl
curl -X POST -F "file=@sample_test.jpg" http://localhost:8080/api/test/image
```

---

## 🌐 คู่มือการ Deploy ไปยัง Production Cloud Server (เว็บไซต์จริง)

> [!IMPORTANT]
> **ข้อกำหนดเรื่องกล้องเว็บแคมบนเบราว์เซอร์ (HTTPS):**
> เบราว์เซอร์สมัยใหม่ (Chrome, Safari, Edge) จะอนุญาตให้เข้าถึงกล้องเว็บแคม (`getUserMedia`) ได้เฉพาะเมื่อเข้าผ่าน **`localhost`** หรือ **`https://`** เท่านั้น
> หากนำไปขึ้น Server จริงที่มี IP สาธารณะ หรือโดเมน เช่น `ai.yourdomain.com` **จำเป็นต้องครอบด้วย HTTPS** (แนะนำให้ใช้ Nginx Reverse Proxy ร่วมกับ Let's Encrypt SSL ฟรี)

### ขั้นตอนที่ 1: นำ Image ขึ้น Container Registry (เลือกอย่างใดอย่างหนึ่ง)

```bash
# 1. Tag Image ไปยัง Docker Hub หรือ GHCR (ตัวอย่าง: dockerhub_username)
docker tag yolo-safety-monitor:latest <YOUR_DOCKERHUB_USERNAME>/yolo-safety-monitor:latest

# 2. Push ขึ้น Registry
docker push <YOUR_DOCKERHUB_USERNAME>/yolo-safety-monitor:latest
```

### ขั้นตอนที่ 2: บน Cloud Server (Ubuntu / Debian VPS)

```bash
# 1. Pull Image และรันด้วยคำสั่งเดียว
docker run -d \
  --name yolo-safety-monitoring \
  -p 8080:8080 \
  --restart unless-stopped \
  --shm-size=2g \
  -v /var/data/yolo/config:/app/config \
  -v /var/data/yolo/results:/app/results \
  <YOUR_DOCKERHUB_USERNAME>/yolo-safety-monitor:latest
```

### ขั้นตอนที่ 3: ตั้งค่า Nginx Reverse Proxy + SSL Certificate (สำหรับ HTTPS)

สร้างไฟล์คอนฟิก `/etc/nginx/sites-available/yolo-safety`:

```nginx
server {
    server_name ai.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # รองรับการอัปโหลดไฟล์วิดีโอขนาดใหญ่
        client_max_body_size 500M;
        proxy_read_timeout 300s;
        proxy_send_timeout 300s;
    }
}
```

เปิดใช้งานและติดตั้ง SSL อัตโนมัติ:

```bash
sudo ln -s /etc/nginx/sites-available/yolo-safety /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d ai.yourdomain.com
```

ระบบพร้อมให้บริการ Production ระดับ Enterprise ทันที! 🚀
