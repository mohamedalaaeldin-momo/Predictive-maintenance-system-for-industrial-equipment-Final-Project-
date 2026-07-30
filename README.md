<div dir="rtl">

# 🛠️ نظام التنبؤ بأعطال الماكينات (Machine Downtime Predictor)

## 1. فكرة المشروع

النموذج المرفق (`machine_downtime_pipeline.pkl`) عبارة عن Pipeline من مكتبة scikit-learn يتكوّن من:

1. **معالجة أولية (Preprocessing):** تعويض القيم المفقودة بالوسيط (`SimpleImputer(strategy="median")`) ثم توحيد المقياس (`StandardScaler`).
2. **نموذج تصنيف:** `RandomForestClassifier` (300 شجرة) يصنّف القراءة إلى إحدى فئتين:
   - `0` → الماكينة تعمل بشكل طبيعي (No Machine Failure)
   - `1` → الماكينة معرّضة للعطل وتحتاج صيانة (Machine Failure)

النموذج يعتمد على **6 حساسات (Features)** بالضبط، بنفس الأسماء والترتيب التالي:

| اسم العمود في النموذج       | الوصف                    | الوحدة |
| ---------------------------- | ------------------------- | ------ |
| `Torque(Nm)`                 | عزم الدوران                | Nm     |
| `Hydraulic_Pressure(bar)`    | الضغط الهيدروليكي          | bar    |
| `Cutting(kN)`                 | قوة القطع                  | kN     |
| `Coolant_Pressure(bar)`      | ضغط سائل التبريد           | bar    |
| `Spindle_Speed(RPM)`         | سرعة دوران المغزل          | RPM    |
| `Coolant_Temperature`        | درجة حرارة سائل التبريد    | °C     |

نطاقات الـ sliders في الواجهة مأخوذة من إحصائيات النموذج نفسه (المتوسط والانحراف المعياري المخزّنين داخل `StandardScaler`)، والقيم الافتراضية هي الوسيط (median) المخزّن داخل `SimpleImputer`، حتى تُطابق البيانات التي دُرِّب عليها النموذج فعليًا.

---

## 2. البنية المعمارية

```
project/
├── backend/                         # FastAPI — خدمة استدلال (Inference API)
│   ├── api/
│   │   └── index.py                 # نقطة الدخول (ASGI app) — تُستخدم محليًا وعلى Vercel
│   ├── model/
│   │   └── machine_downtime_pipeline.pkl
│   ├── requirements.txt
│   └── vercel.json
│
├── frontend/                        # Next.js (App Router) — لوحة تحكم تفاعلية
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx                 # الواجهة الرئيسية: sliders + عداد النتيجة
│   │   └── globals.css
│   ├── package.json
│   ├── tailwind.config.ts
│   └── .env.local.example
│
└── README.md
```

**لماذا مشروعان منفصلان على Vercel؟**
Vercel يتعامل بشكل أفضل وأكثر استقرارًا عندما يكون كل Runtime (Python للـ Backend، Node.js للـ Frontend) في **مشروع Vercel مستقل** له نفس الـ Repository لكن بـ Root Directory مختلف (`backend` و `frontend`)، بدلاً من محاولة دمجهما في `vercel.json` واحد. هذا هو الأسلوب الموصى به من Vercel نفسها بدون استخدام Docker، وهو ما يعتمده هذا المشروع.

### تدفّق العمل (Flow)

```
المستخدم يحرّك الـ sliders على الواجهة
        │
        ▼
Next.js Frontend  → POST /predict → FastAPI Backend
        │                                  │
        │                                  ▼
        │                    تحميل model.pkl مرة واحدة عند الإقلاع
        │                    (Cold Start) ثم model.predict_proba()
        │                                  │
        ◄──────────── JSON بالنتيجة ───────┘
        ▼
عرض النتيجة على عداد دائري (Gauge) + حالة نصية
```

---

## 3. تشغيل المشروع محليًا

### أ) تشغيل الـ Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate      # على Windows: venv\Scripts\activate

pip install -r requirements.txt

uvicorn api.index:app --reload --port 8000
```

بعد التشغيل، يمكن التأكد من عمل الخدمة عبر:

- `GET http://localhost:8000/health` → للتأكد من تحميل النموذج بنجاح
- `POST http://localhost:8000/predict` → لإرسال قراءة واختبار التنبؤ، مثال للـ body:

```json
{
  "Torque(Nm)": 25.2,
  "Hydraulic_Pressure(bar)": 101.4,
  "Cutting(kN)": 2.79,
  "Coolant_Pressure(bar)": 4.94,
  "Spindle_Speed(RPM)": 20283,
  "Coolant_Temperature": 18.6
}
```

### ب) تشغيل الـ Frontend

```bash
cd frontend
npm install

cp .env.local.example .env.local
# تأكد أن NEXT_PUBLIC_API_URL=http://localhost:8000

npm run dev
```

ثم افتح `http://localhost:3000` في المتصفح.

---

## 4. ⚠️ ملاحظة مهمة جدًا حول توافق إصدار scikit-learn

أثناء تجهيز هذا المشروع، تبيّن عمليًا أن ملف `machine_downtime_pipeline.pkl` تم تدريبه وحفظه باستخدام **scikit-learn 1.6.1**. عند محاولة تحميله (`pickle.load` / `joblib.load`) باستخدام إصدار أحدث (تم اختباره على 1.8.0)، يحدث خطأ داخلي أثناء إعادة بناء الكائنات الداخلية لـ `ColumnTransformer` و `SimpleImputer` (رسائل من نوع `AttributeError` تتعلق بخصائص داخلية غير موجودة في الإصدار الأحدث)، رغم أن التحميل قد يبدو ناجحًا في البداية.

**لذلك تم تثبيت (pin) الإصدار بدقة في `backend/requirements.txt`:**

```
scikit-learn==1.6.1
```

لا تُغيّر هذا الإصدار إلا إذا أعدت تدريب النموذج وحفظه من جديد بالإصدار الجديد. هذه الملاحظة توفر عليك وقتًا كبيرًا في تشخيص أخطاء غامضة عند النشر.

---

## 5. النشر على Vercel (بدون Docker)

### أ) نشر الـ Backend (FastAPI)

1. ادفع المجلد الكامل `project/` إلى مستودع GitHub.
2. من لوحة تحكم Vercel: **Add New → Project** → اختر نفس الـ Repository.
3. في إعداد **Root Directory** اختر `backend`.
4. Vercel يكتشف تلقائيًا وجود `api/index.py` وملف `requirements.txt` وينشئ Serverless Function بـ Python runtime — لا حاجة لأي إعداد إضافي غير الموجود في `vercel.json`.
5. بعد اكتمال النشر، انسخ رابط المشروع (مثال: `https://your-backend.vercel.app`).
6. من **Project Settings → Environment Variables** أضف:
   ```
   FRONTEND_ORIGIN = https://your-frontend.vercel.app
   ```
   (سيُستخدم هذا لضبط CORS بشكل صحيح — أضِفه بعد نشر الـ Frontend في الخطوة التالية وأعد النشر Redeploy).

### ب) نشر الـ Frontend (Next.js)

1. من لوحة تحكم Vercel مرة أخرى: **Add New → Project** → اختر نفس الـ Repository.
2. في **Root Directory** اختر `frontend` (Vercel يكتشف Next.js تلقائيًا).
3. من **Environment Variables** أضف:
   ```
   NEXT_PUBLIC_API_URL = https://your-backend.vercel.app
   ```
4. اضغط **Deploy**.
5. بعد نجاح النشر، ارجع لمشروع الـ Backend وحدّث `FRONTEND_ORIGIN` بالرابط النهائي للـ Frontend، ثم اعمل Redeploy للـ Backend حتى يعمل CORS بشكل صحيح.

بهذا يكون لديك رابطان منفصلان (Backend / Frontend) يتواصلان معًا عبر HTTPS بدون أي حاجة لـ Docker أو خادم مُدار يدويًا.

---

## 6. تفاصيل الـ API

### `POST /predict`

**Request body** (كل الحقول أرقام `float`، بنفس أسماء أعمدة النموذج):

```json
{
  "Torque(Nm)": number,
  "Hydraulic_Pressure(bar)": number,
  "Cutting(kN)": number,
  "Coolant_Pressure(bar)": number,
  "Spindle_Speed(RPM)": number,
  "Coolant_Temperature": number
}
```

**Response:**

```json
{
  "prediction": "Machine_Failure | No_Machine_Failure",
  "needs_maintenance": true,
  "probability_failure": 0.7833,
  "confidence": 0.7833,
  "needs_manual_review": false
}
```

`needs_manual_review` تكون `true` عندما تكون ثقة النموذج في تنبؤه أقل من 60%، للإشارة إلى ضرورة مراجعة بشرية للقراءة بدلاً من الاعتماد الكامل على النموذج.

---

## 7. التقنيات المستخدمة

- **Backend:** FastAPI, scikit-learn, pandas, Pydantic v2
- **Frontend:** Next.js 14 (App Router), TypeScript, Tailwind CSS
- **النشر:** Vercel (مشروعان منفصلان، بدون Docker)

</div>
