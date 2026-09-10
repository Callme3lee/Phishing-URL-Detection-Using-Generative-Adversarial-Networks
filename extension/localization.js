(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else {
    root.PhishGuardI18n = api;
    if (typeof document !== "undefined") api.initialize();
  }
})(globalThis, function () {
  "use strict";

  const STORAGE_KEY = "uiLanguage";
  const ARABIC = {
    "Phishing URL Guard": "حارس روابط التصيد",
    "Phishing URL Guard — Recall Trial": "حارس روابط التصيد — تجربة الاستدعاء",
    "LOCAL WEBSITE CHECK": "فحص محلي للموقع",
    "Checks happen privately on this device": "يتم الفحص بخصوصية على هذا الجهاز",
    "How warnings behave": "طريقة عرض التحذيرات",
    "Normal mode pauses only the highest-risk URLs.": "الوضع العادي يوقف مؤقتاً الروابط الأعلى خطورة فقط.",
    "Warnings only shows alerts but never pauses a page.": "وضع التحذيرات فقط يعرض التنبيهات من دون إيقاف الصفحة.",
    "Navigation mode": "وضع التصفح",
    "Block high risk": "إيقاف الخطر العالي",
    "Warnings only": "تحذيرات فقط",
    "Current website": "الموقع الحالي",
    "Reading current tab…": "جارٍ قراءة علامة التبويب…",
    "Loading current URL…": "جارٍ تحميل الرابط…",
    "SITE CHECK": "نتيجة الفحص",
    "Ready to check": "جاهز للفحص",
    "Select Check this page to examine its web address.": "اختر «افحص هذه الصفحة» لتحليل عنوانها.",
    "RISK SCORE": "درجة الخطورة",
    "Check this page": "افحص هذه الصفحة",
    "Checking privately…": "جارٍ الفحص بخصوصية…",
    "Always allow this exact website": "السماح دائماً لهذا الموقع بالضبط",
    "Always allow": "السماح دائماً لـ",
    "this website": "هذا الموقع",
    "Only this exact website name will be allowed:": "سيُسمح لاسم الموقع المحدد هذا فقط:",
    "You can remove it later. If local feedback is on, this also saves the warning as incorrect.": "يمكنك إزالة السماح لاحقاً. وإذا كانت الملاحظات المحلية مفعلة، فسيُحفظ التحذير أيضاً على أنه غير صحيح.",
    "Future pages on this exact website name will not show URL warnings. You can remove it later. If local feedback is on, this also saves the warning as incorrect.": "لن تظهر تحذيرات الروابط مستقبلاً لهذا الاسم المحدد فقط، ويمكنك إزالة السماح لاحقاً. وإذا كانت الملاحظات المحلية مفعلة، فسيُحفظ التحذير أيضاً على أنه غير صحيح.",
    "Remove this website from my allowed list": "إزالة هذا الموقع من قائمة السماح",
    "Was this result correct?": "هل كانت النتيجة صحيحة؟",
    "Checking feedback settings…": "جارٍ التحقق من إعدادات الملاحظات…",
    "Activity": "النشاط",
    "This warning was incorrect": "هذا التحذير غير صحيح",
    "This phishing page was missed": "لم يكتشف صفحة التصيد هذه",
    "Feedback is stored on this device only.": "تُحفظ الملاحظات على هذا الجهاز فقط.",
    "About this result": "عن هذه النتيجة",
    "Detailed result information will appear after scanning.": "ستظهر تفاصيل النتيجة بعد الفحص.",
    "HYB-MS + 10% GAN · three-seed local ensemble": "HYB-MS + 10% GAN · تجميع محلي لثلاث بذور",
    "The score ranks URL-text suspicion. It is not a trust percentage or proof that a site is malicious.": "ترتّب الدرجة مدى الاشتباه في نص الرابط، وهي ليست نسبة ثقة ولا دليلاً قاطعاً على أن الموقع ضار.",
    "Try the safe URL checker": "جرّب فاحص الروابط الآمن",
    "ALLOW THIS EXACT WEBSITE": "السماح لهذا الموقع بالضبط",
    "Always allow this website on this device?": "هل تريد السماح دائماً لهذا الموقع على هذا الجهاز؟",
    "This website received a warning. If you allow it, future pages on this exact website name will not show URL warnings or pauses.": "تلقى هذا الموقع تحذيراً. إذا سمحت له، فلن تعرض الصفحات المستقبلية على اسم الموقع نفسه تحذيرات أو إيقافاً مؤقتاً.",
    "This affects only the exact name shown above. It does not allow look-alike websites or other subdomains.": "ينطبق هذا على الاسم المعروض أعلاه فقط، ولا يسمح بالمواقع المشابهة أو النطاقات الفرعية الأخرى.",
    "I checked the spelling and want to always allow this exact website.": "تحققت من كتابة الاسم وأريد السماح دائماً لهذا الموقع المحدد.",
    "Cancel": "إلغاء",
    "Always allow website": "السماح دائماً بالموقع",
    "URL text only": "نص الرابط فقط",
    "Optional feedback": "ملاحظات اختيارية",
    "No automatic upload": "لا يوجد رفع تلقائي",
    "Potential phishing URL · Phishing URL Guard": "رابط تصيد محتمل · حارس روابط التصيد",
    "On-device URL protection": "حماية الروابط على الجهاز",
    "Processed locally": "تمت المعالجة محلياً",
    "NAVIGATION PAUSED": "تم إيقاف التصفح مؤقتاً",
    "This URL may be dangerous": "قد يكون هذا الرابط خطيراً",
    "The local detector placed this destination at its highest risk level. Going back is the safest choice.": "صنّف الفاحص المحلي هذه الوجهة ضمن أعلى مستوى خطورة. الرجوع هو الخيار الأكثر أماناً.",
    "The local detector placed this destination at its highest": "صنّف الفاحص المحلي هذه الوجهة ضمن أعلى مستوى",
    "risk": "خطورة",
    "level. Going back is the safest choice.": "للخطورة. الرجوع هو الخيار الأكثر أماناً.",
    "Recommended action": "الإجراء الموصى به",
    "Return to the previous page unless you recognize and intentionally entered this exact destination.": "ارجع إلى الصفحة السابقة ما لم تكن تعرف هذه الوجهة وأدخلتها عمداً.",
    "Destination under review": "الوجهة قيد المراجعة",
    "Very high risk": "خطورة عالية جداً",
    "Unknown host": "اسم موقع غير معروف",
    "URL risk score": "درجة خطورة الرابط",
    "Go back to safety": "الرجوع بأمان",
    "Continue once (do not allow)": "المتابعة مرة واحدة دون سماح دائم",
    "Know and trust this website?": "هل تعرف هذا الموقع وتثق به؟",
    "Allow future pages on this exact website name. You can remove it later. If local feedback is on, this also saves the warning as incorrect.": "اسمح بالصفحات المستقبلية على اسم هذا الموقع المحدد فقط، ويمكنك إزالة السماح لاحقاً. وإذا كانت الملاحظات المحلية مفعلة، فسيُحفظ التحذير أيضاً على أنه غير صحيح.",
    "I know this website": "أنا أعرف هذا الموقع",
    "Use permanent allow only when you have checked the exact website name. You can remove it later from the extension popup.": "استخدم السماح الدائم فقط بعد التحقق من اسم الموقع بدقة. ويمكنك إزالته لاحقاً من نافذة الإضافة.",
    "Always allow this exact website on this device": "السماح دائماً لهذا الموقع المحدد على هذا الجهاز",
    "What this warning means": "ماذا يعني هذا التحذير",
    "The detector found a strong pattern in the URL text. It does not inspect the page itself, so this warning is not proof of phishing.": "وجد الفاحص نمطاً قوياً في نص الرابط، لكنه لا يفحص محتوى الصفحة؛ لذلك فالتحذير ليس دليلاً قاطعاً على التصيد.",
    "Report this as a false alarm": "الإبلاغ عن تحذير غير صحيح",
    "PERMANENT LOCAL EXCEPTION": "استثناء محلي دائم",
    "Confirm the exact website before allowing it": "تحقق من اسم الموقع قبل السماح له",
    "Allowing this exact website name will stop future URL warnings and navigation pauses for it on this device.": "السماح لاسم هذا الموقع المحدد سيوقف تحذيراته وإيقاف التصفح له مستقبلاً على هذا الجهاز.",
    "Only continue if you recognise the website, its spelling is correct, and you intended to visit it. Otherwise, cancel and go back.": "تابع فقط إذا كنت تعرف الموقع، وتأكدت من كتابة اسمه، وكنت تقصد زيارته. خلاف ذلك ألغِ العملية وارجع.",
    "I checked the exact website name and accept the risk of allowing it permanently.": "تحققت من اسم الموقع بدقة وأقبل مخاطر السماح الدائم له.",
    "Always allow and continue": "السماح دائماً والمتابعة",
    "Local URL protection · Only the highest-risk URLs pause navigation": "حماية محلية للروابط · يتم إيقاف الروابط الأعلى خطورة فقط",
    "Exercise caution": "توخَّ الحذر",
    "CAUTION ADVISED": "ينصح بالحذر",
    "LOCAL CHECK": "فحص محلي",
    "Exercise caution on this site": "توخَّ الحذر في هذا الموقع",
    "Current destination": "الوجهة الحالية",
    "The URL has a suspicious text pattern.": "يحتوي نص الرابط على نمط مشبوه.",
    "Open the extension to see why this URL received this result.": "افتح الإضافة لمعرفة سبب حصول هذا الرابط على هذه النتيجة.",
    "URL check · not proof of phishing": "فحص الرابط · ليس دليلاً قاطعاً على التصيد",
    "Dismiss this caution": "إغلاق هذا التنبيه",
    "Always allow this website": "السماح دائماً لهذا الموقع",
    "Allowing…": "جارٍ السماح…",
    "Exact website only · saves an incorrect-warning report locally when feedback is on": "للموقع المحدد فقط · يحفظ تقريراً محلياً عن التحذير غير الصحيح عند تفعيل الملاحظات",
    "REVIEW SUGGESTED": "ينصح بالمراجعة",
    "HIGH-RISK PATTERN": "نمط عالي الخطورة",
    "VERY HIGH-RISK PATTERN": "نمط شديد الخطورة",
    "PUBLIC IP REVIEW": "مراجعة عنوان IP عام",
    "Detection Activity · Phishing URL Guard": "نشاط الاكتشاف · حارس روابط التصيد",
    "LOCAL DETECTION ACTIVITY · v0.6.0": "نشاط الاكتشاف المحلي · v0.6.0",
    "Your detection activity": "نشاط الاكتشاف لديك",
    "Choose whether to keep local activity counts and send feedback about incorrect warnings. Nothing is uploaded automatically.": "اختر ما إذا كنت تريد حفظ إحصاءات النشاط محلياً وإرسال ملاحظات عن التحذيرات غير الصحيحة. لا يتم رفع أي شيء تلقائياً.",
    "Your choice": "اختيارك",
    "Turn on local detection history?": "هل تريد تشغيل سجل الاكتشاف المحلي؟",
    "The protection works either way. Turning this on saves local activity counts and enables the feedback buttons.": "تعمل الحماية في الحالتين. يؤدي تشغيل السجل إلى حفظ الإحصاءات محلياً وتفعيل أزرار الملاحظات.",
    "URLs are checked on this device. Ordinary visited URLs are": "يتم فحص الروابط على هذا الجهاز. الروابط العادية التي تزورها",
    "not saved as readable text": "لا تُحفظ كنص مقروء",
    "A private one-way identifier is kept locally so repeated checks of the same URL are counted once. It is not included in exports.": "يُحفظ معرّف خاص أحادي الاتجاه محلياً حتى يُحسب الفحص المتكرر للرابط مرة واحدة. ولا يُضمّن في الملفات المصدّرة.",
    "The extension saves local totals for checked URLs, results, ignored repeats, errors, and checking time.": "تحفظ الإضافة محلياً إجماليات الروابط المفحوصة والنتائج والتكرارات المتجاهلة والأخطاء ووقت الفحص.",
    "A readable URL is saved only when you press a feedback button. Login details, query values, and page fragments are removed.": "لا يُحفظ رابط مقروء إلا عند الضغط على زر للملاحظات، وتُحذف تفاصيل الدخول وقيم الاستعلام وأجزاء الصفحة.",
    "Reported paths can still contain personal information, so review an exported file before sharing it.": "قد تحتوي المسارات المبلغ عنها على معلومات شخصية، لذلك راجع الملف المصدّر قبل مشاركته.",
    "You choose if and when to export the results. You can pause or erase the local history at any time.": "أنت تختار ما إذا كنت تريد تصدير النتائج ومتى. ويمكنك إيقاف السجل المحلي أو مسحه في أي وقت.",
    "Turn on local history": "تشغيل السجل المحلي",
    "No thanks": "لا، شكراً",
    "Local history is on": "السجل المحلي مفعّل",
    "Local history is paused": "السجل المحلي متوقف مؤقتاً",
    "Pause history": "إيقاف النسجيل مؤقتاً",
    "Resume history": "استئناف التسجيل",
    "Detection summary": "ملخص الاكتشاف",
    "Your activity": "نشاطك",
    "Activity ID": "معرّف النشاط",
    "Earlier results were kept separately in the detailed JSON export. The numbers below belong to the current detection version.": "حُفظت النتائج السابقة بصورة منفصلة في ملف JSON التفصيلي. الأرقام أدناه تخص إصدار الاكتشاف الحالي.",
    "Unique URLs checked": "الروابط الفريدة التي تم فحصها",
    "Total checks performed": "إجمالي عمليات الفحص",
    "Repeat checks ignored": "عمليات الفحص المتكررة المتجاهلة",
    "Incorrect warnings reported": "التحذيرات غير الصحيحة المبلغ عنها",
    "Missed threats reported": "التهديدات الفائتة المبلغ عنها",
    "Average check time": "متوسط وقت الفحص",
    "Results for unique URLs checked": "نتائج الروابط الفريدة التي تم فحصها",
    "Save a copy": "حفظ نسخة",
    "Export your activity": "تصدير نشاطك",
    "The detailed JSON includes activity counts and only the URLs you deliberately reported. The CSV contains summary totals. Private identifiers used to ignore repeat checks are never exported.": "يتضمن ملف JSON التفصيلي إحصاءات النشاط والروابط التي أبلغت عنها عمداً فقط. ويحتوي ملف CSV على الإجماليات. لا يتم تصدير المعرّفات الخاصة المستخدمة لتجاهل التكرار.",
    "Export detailed results": "تصدير النتائج التفصيلية",
    "Export summary": "تصدير الملخص",
    "Erase local activity": "مسح النشاط المحلي",
    "This permanently clears activity counts, private repeat-check data, and feedback saved by the extension.": "يمسح هذا نهائياً إحصاءات النشاط وبيانات منع التكرار والملاحظات المحفوظة في الإضافة.",
    "Erase activity": "مسح النشاط",
    "Local URL protection · optional feedback · no automatic upload": "حماية محلية للروابط · ملاحظات اختيارية · لا يوجد رفع تلقائي",
    "Phishing URL Guard — Safe Demo": "حارس روابط التصيد — الفاحص الآمن",
    "URL risk detection": "اكتشاف مخاطر الروابط",
    "Local processing": "معالجة محلية",
    "Safe test environment": "بيئة اختبار آمنة",
    "Inspect a URL without opening the website.": "افحص الرابط من دون فتح الموقع.",
    "The local detector checks the text and structure of the URL you enter. It never opens or contacts the destination.": "يفحص الكاشف المحلي نص الرابط وبنيته من دون فتح الوجهة أو الاتصال بها.",
    "Local ONNX model": "نموذج ONNX محلي",
    "No navigation": "دون فتح الموقع",
    "URL scanner": "فاحص الروابط",
    "Check a destination": "افحص وجهة",
    "Offline checker": "فاحص دون اتصال",
    "Paste or type a full URL": "الصق رابطاً كاملاً أو اكتبه",
    "Score URL": "افحص الرابط",
    "Scoring…": "جارٍ الفحص…",
    "The URL stays on this device.": "يبقى الرابط على هذا الجهاز.",
    "Detection result": "نتيجة الاكتشاف",
    "Waiting for a URL": "بانتظار رابط",
    "Enter a URL above to view its lexical risk score.": "أدخل رابطاً أعلاه لعرض درجة خطورته النصية.",
    "Check status": "حالة الفحص",
    "Ready": "جاهز",
    "Unavailable": "غير متاح",
    "Local history is off. Open Activity to turn it on.": "السجل المحلي متوقف. افتح صفحة النشاط لتشغيله.",
    "How to read the result": "كيفية قراءة النتيجة",
    "A warning, not a final verdict": "تحذير وليس حكماً نهائياً",
    "A high score means the URL text resembles patterns learned from phishing examples.": "تعني الدرجة المرتفعة أن نص الرابط يشبه أنماطاً تعلمها النموذج من أمثلة التصيد.",
    "Lexical patterns": "الأنماط النصية",
    "Length, symbols, tokens and character sequences": "الطول والرموز والكلمات وتسلسل المحارف",
    "Hybrid model": "النموذج الهجين",
    "Character CNN combined with engineered URL features": "شبكة CNN للمحارف مع خصائص الرابط المصممة",
    "Use caution": "توخَّ الحذر",
    "The checker can warn incorrectly or miss a dangerous URL": "قد يصدر الفاحص تحذيراً خاطئاً أو يفوّت رابطاً خطيراً",
    "Not checked in this demo": "لا يتم فحصه في هذا العرض",
    "DNS · page content · certificates · redirects · reputation": "DNS · محتوى الصفحة · الشهادات · إعادة التوجيه · السمعة",
    "Built-in test cases": "حالات اختبار مدمجة",
    "Try an example safely": "جرّب مثالاً بأمان",
    "Examples are scored as text and are never opened.": "تُفحص الأمثلة كنصوص ولا يتم فتحها.",
    "Local URL checker": "فاحص روابط محلي",
    "On-device analysis · v0.6.0": "تحليل على الجهاز · v0.6.0",
    "Ordinary reserved domain": "نطاق محجوز عادي",
    "Redirect-style query": "استعلام يشبه إعادة التوجيه",
    "Hosted login path": "مسار تسجيل دخول مستضاف",
    "Encoded payment wording": "عبارات دفع مشفّرة",
    "Reserved IP login pattern": "نمط دخول بعنوان IP محجوز",
    "Unable to check URL": "تعذر فحص الرابط",
    "The URL check failed.": "فشل فحص الرابط.",
    "Extension unavailable": "الإضافة غير متاحة",
    "No warning found": "لم يتم العثور على تحذير",
    "The detector did not find a strong phishing pattern in this URL.": "لم يجد الفاحص نمط تصيد قوياً في هذا الرابط.",
    "Review suggested": "ينصح بالمراجعة",
    "This URL looks unusual, but there is not enough evidence to call it phishing. You may continue.": "يبدو الرابط غير معتاد، لكن الأدلة غير كافية لاعتباره تصيداً. يمكنك المتابعة بحذر.",
    "Some parts of this URL look suspicious. This does not prove the site is phishing, and the page remains available.": "تبدو بعض أجزاء الرابط مشبوهة. هذا لا يثبت أن الموقع للتصيد، وتبقى الصفحة متاحة.",
    "High-risk URL": "رابط عالي الخطورة",
    "This URL strongly resembles phishing patterns. Navigation is paused, but you can choose to continue.": "يشبه هذا الرابط أنماط التصيد بدرجة كبيرة. تم إيقاف التصفح مؤقتاً، ويمكنك اختيار المتابعة.",
    "This URL strongly resembles phishing patterns. Avoid entering passwords or payment information unless you trust the destination.": "يشبه هذا الرابط أنماط التصيد بدرجة كبيرة. تجنب إدخال كلمات المرور أو معلومات الدفع ما لم تثق بالوجهة.",
    "Very high-risk URL": "رابط شديد الخطورة",
    "This URL reached the highest risk level. Navigation is paused, but you can choose to continue.": "وصل الرابط إلى أعلى مستوى خطورة. تم إيقاف التصفح مؤقتاً، ويمكنك اختيار المتابعة.",
    "Allowed by you": "مسموح بواسطتك",
    "You added this exact website name to your allowed-sites list. The detector still records its result but will not interrupt the page.": "أضفت اسم هذا الموقع المحدد إلى قائمة السماح. يستمر الفاحص في تسجيل النتيجة لكنه لن يوقف الصفحة.",
    "Recognised service": "خدمة موثوقة تم التعرّف عليها",
    "This address matches a reviewed official service page, so no URL-text warning is shown. Links and redirect destinations are checked separately.": "يطابق هذا العنوان صفحة خدمة رسمية تمت مراجعتها، لذلك لا يظهر تحذير لنص الرابط. تُفحص الروابط ووجهات إعادة التوجيه بصورة منفصلة.",
    "Recognised shared content": "محتوى مشترك معروف",
    "This page is hosted by Google, and the URL model did not find a strong phishing pattern.": "هذه الصفحة مستضافة لدى Google، ولم يجد نموذج الرابط نمط تصيد قويًا.",
    "Shared content — review suggested": "محتوى مشترك — يُنصح بالمراجعة",
    "This page is hosted by Google but was created or shared by a user. Verify who sent it before entering sensitive information.": "هذه الصفحة مستضافة لدى Google لكنها أُنشئت أو شُوركت بواسطة مستخدم. تحقق من المرسل قبل إدخال معلومات حساسة.",
    "Shared content — use caution": "محتوى مشترك — استخدم الحذر",
    "This user-created Google link has a high URL risk score. The page remains available, but verify the sender before continuing.": "رابط Google هذا أنشأه مستخدم ولديه درجة خطورة مرتفعة. تبقى الصفحة متاحة، لكن تحقق من المرسل قبل المتابعة.",
    "Shared content — high risk": "محتوى مشترك — خطورة عالية",
    "Shared content — very high risk": "محتوى مشترك — خطورة شديدة",
    "This user-created Google link has a high URL risk score. Navigation is paused in Normal mode, but you can choose to continue.": "رابط Google هذا أنشأه مستخدم ولديه درجة خطورة عالية. تم إيقاف التصفح في الوضع العادي، ويمكنك اختيار المتابعة.",
    "This user-created Google link reached the highest URL risk range. Navigation is paused in Normal mode, but you can choose to continue.": "وصل رابط Google الذي أنشأه مستخدم إلى أعلى نطاق للخطورة. تم إيقاف التصفح في الوضع العادي، ويمكنك اختيار المتابعة.",
    "This user-created Google link reached the highest URL risk range. It is not blocked automatically, but avoid entering passwords or payment information.": "وصل رابط Google الذي أنشأه مستخدم إلى أعلى نطاق للخطورة. لن يُحظر تلقائيًا، لكن تجنب إدخال كلمات المرور أو معلومات الدفع.",
    "Local network address": "عنوان شبكة محلية",
    "This looks like a valid address on your local network. It is shown for review but is not blocked using the URL-text score alone.": "يبدو هذا عنواناً صالحاً على شبكتك المحلية. يظهر للمراجعة ولا يُحظر اعتماداً على درجة نص الرابط وحدها.",
    "Public IP address": "عنوان IP عام",
    "This is a valid public IP address. Because it has no website name for the detector to examine, it is shown for review rather than blocked automatically.": "هذا عنوان IP عام صالح. ولأنه لا يحتوي على اسم موقع يمكن فحصه، فإنه يظهر للمراجعة بدلاً من حظره تلقائياً.",
    "URL review unavailable": "مراجعة الرابط غير متاحة",
    "The model returned an unknown policy tier.": "أعاد النموذج مستوى سياسة غير معروف.",
    "Very high-risk pattern — continued once": "نمط شديد الخطورة — متابعة لمرة واحدة",
    "Very high-risk pattern — passive mode": "نمط شديد الخطورة — وضع التحذيرات فقط",
    "High-risk pattern — continued once": "نمط عالي الخطورة — متابعة لمرة واحدة",
    "High-risk pattern — passive mode": "نمط عالي الخطورة — وضع التحذيرات فقط",
    "You chose to continue once. The warning remains visible, so avoid entering sensitive information unless you trust the destination.": "اخترت المتابعة لمرة واحدة. يبقى التحذير ظاهراً، فتجنب إدخال معلومات حساسة ما لم تثق بالوجهة.",
    "Passive mode keeps this high-risk shared page available and shows a warning instead of interrupting navigation.": "يبقي وضع التحذيرات فقط هذه الصفحة المشتركة عالية الخطورة متاحة ويعرض تنبيهاً بدلاً من إيقاف التصفح.",
    "You chose to continue once. The warning remains active, so avoid entering passwords, payment details, or other sensitive information.": "اخترت المتابعة لمرة واحدة. يبقى التحذير فعالاً، فتجنب إدخال كلمات المرور أو تفاصيل الدفع أو المعلومات الحساسة.",
    "The URL reached the highest risk level, but Passive mode shows the warning without interrupting navigation.": "وصل الرابط إلى أعلى مستوى خطورة، لكن وضع التحذيرات فقط يعرض التنبيه من دون إيقاف التصفح.",
    "{count} unique URLs · {reports} reports": "{count} رابط فريد · {reports} بلاغ",
    "{milliseconds} ms check time · processed on this device": "وقت الفحص {milliseconds} مللي ثانية · تمت المعالجة على هذا الجهاز",
    "Risk score {score}": "درجة الخطورة {score}",
    "Local detection history is on.": "سجل الاكتشاف المحلي مفعّل.",
    "Local history is off. URL protection still works.": "السجل المحلي متوقف. ما زالت حماية الروابط تعمل.",
    "Unable to read local activity.": "تعذرت قراءة النشاط المحلي.",
    "Unable to update local history settings.": "تعذر تحديث إعدادات السجل المحلي.",
    "Local history is full. Export and erase it to begin a new record. URL protection continues.": "السجل المحلي ممتلئ. صدّره ثم امسحه لبدء سجل جديد. تستمر حماية الروابط.",
    "Erase all local activity counts, repeat-check data, and feedback reports? This cannot be undone.": "هل تريد مسح جميع إحصاءات النشاط وبيانات التكرار وبلاغات الملاحظات؟ لا يمكن التراجع عن ذلك."
    ,"The website could not be added to your allowed list.": "تعذرت إضافة الموقع إلى قائمة السماح."
    ,"Saved locally. Query values and fragments were removed.": "تم الحفظ محلياً بعد إزالة قيم الاستعلام وأجزاء الصفحة."
    ,"The report could not be saved.": "تعذر حفظ البلاغ."
    ,"allowed by you": "مسموح بواسطتك"
    ,"no warning": "دون تحذير"
    ,"local IP": "عنوان IP محلي"
    ,"review suggested": "ينصح بالمراجعة"
    ,"public IP review": "مراجعة عنوان IP عام"
    ,"caution": "حذر"
    ,"warning": "تحذير"
    ,"block": "إيقاف"
    ,"high risk": "خطورة عالية"
    ,"very high risk": "خطورة عالية جداً"
    ,"not available": "غير متاح"
    ,"{paused} pages paused · {submissions} feedback submissions · {evicted} older reports removed after reaching the storage limit": "{paused} صفحة موقوفة · {submissions} ملاحظة مرسلة · {evicted} بلاغ قديم حُذف بعد بلوغ حد التخزين"
    ,"{services} recognised service pages allowed quietly · {checks} model checks completed": "{services} صفحة خدمة معروفة سُمح بها بهدوء · {checks} عملية فحص مكتملة للنموذج"
    ,"Signals found in this URL": "إشارات موجودة في هذا الرابط"
    ,"These signals are diagnostic clues from model inputs, not proof that the website is phishing.": "هذه الإشارات دلائل تشخيصية من مدخلات النموذج، وليست دليلاً قاطعاً على أن الموقع تصيد."
    ,"Suspicious words: {words}": "كلمات مثيرة للاشتباه: {words}"
    ,"The URL is unusually long ({count} characters)": "الرابط طويل بشكل غير معتاد ({count} حرفاً)"
    ,"The URL contains an unusually high number of symbols": "يحتوي الرابط على عدد كبير وغير معتاد من الرموز"
    ,"The URL or website name contains many digits": "يحتوي الرابط أو اسم الموقع على أرقام كثيرة"
    ,"The website uses several subdomain levels ({count})": "يستخدم الموقع عدة مستويات من النطاقات الفرعية ({count})"
    ,"The URL has a deep path structure ({count} parts)": "يحتوي الرابط على مسار عميق ({count} أجزاء)"
    ,"The query section is unusually long or complex ({count} characters)": "قسم الاستعلام طويل أو معقد بشكل غير معتاد ({count} حرفاً)"
    ,"The website name has a random-looking character pattern": "يحتوي اسم الموقع على نمط أحرف يبدو عشوائياً"
    ,"The URL contains encoded characters": "يحتوي الرابط على أحرف مرمزة"
    ,"The URL uses an explicit port number": "يستخدم الرابط رقم منفذ صريحاً"
    ,"The website name uses Punycode": "يستخدم اسم الموقع ترميز Punycode"
    ,"The URL uses an IP address instead of a website name": "يستخدم الرابط عنوان IP بدلاً من اسم موقع"
    ,"Repeated characters stand out in the URL": "توجد أحرف متكررة بشكل بارز في الرابط"
    ,"The website name has no recognised public suffix": "لا يحتوي اسم الموقع على لاحقة عامة معروفة"
    ,"Query values increased the model score by {difference}": "رفعت قيم الاستعلام درجة النموذج بمقدار {difference}"
    ,"The combined URL pattern was unusual, but no single lexical signal stood out.": "كان النمط العام للرابط غير معتاد، لكن لم تبرز إشارة لغوية واحدة بوضوح."
  };

  const SIGNAL_MESSAGES = {
    suspicious_terms: "Suspicious words: {words}",
    long_url: "The URL is unusually long ({count} characters)",
    many_symbols: "The URL contains an unusually high number of symbols",
    many_digits: "The URL or website name contains many digits",
    deep_subdomains: "The website uses several subdomain levels ({count})",
    deep_path: "The URL has a deep path structure ({count} parts)",
    complex_query: "The query section is unusually long or complex ({count} characters)",
    hostname_entropy: "The website name has a random-looking character pattern",
    encoded_characters: "The URL contains encoded characters",
    explicit_port: "The URL uses an explicit port number",
    punycode_hostname: "The website name uses Punycode",
    ip_hostname: "The URL uses an IP address instead of a website name",
    repeated_characters: "Repeated characters stand out in the URL",
    unrecognized_suffix: "The website name has no recognised public suffix",
    query_increased_score: "Query values increased the model score by {difference}",
  };

  let language = "en";

  function interpolate(template, values = {}) {
    return template.replace(/\{(\w+)\}/g, (_, key) => values[key] ?? `{${key}}`);
  }

  function dynamicArabic(value) {
    let match = value.match(/^(\d+) unique URLs · (\d+) reports$/);
    if (match) return `${match[1]} رابط فريد · ${match[2]} بلاغ`;
    match = value.match(/^([\d.]+) ms check time · processed on this device$/);
    if (match) return `وقت الفحص ${match[1]} مللي ثانية · تمت المعالجة على هذا الجهاز`;
    match = value.match(/^Risk score (.+)$/);
    if (match) return `درجة الخطورة ${match[1]}`;
    if (value.startsWith("Exercise caution. ")) {
      return `توخَّ الحذر. ${translate(value.slice(18), "ar")}`;
    }
    return null;
  }

  function templateArabic(rendered) {
    for (const [source, translated] of Object.entries(ARABIC)) {
      const keys = [...source.matchAll(/\{(\w+)\}/g)].map((match) => match[1]);
      if (!keys.length) continue;
      const pattern = source.split(/\{\w+\}/).map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("(.+?)");
      const match = rendered.match(new RegExp(`^${pattern}$`));
      if (!match) continue;
      return interpolate(translated, Object.fromEntries(keys.map((key, index) => [key, match[index + 1]])));
    }
    return null;
  }

  function translate(value, requestedLanguage = language, values = {}) {
    const source = String(value);
    const rendered = interpolate(source, values);
    if (requestedLanguage !== "ar") return rendered;
    return ARABIC[source] ? interpolate(ARABIC[source], values)
      : (ARABIC[rendered] ?? templateArabic(rendered) ?? dynamicArabic(rendered) ?? rendered);
  }

  function signalText(signal, requestedLanguage = language) {
    const template = SIGNAL_MESSAGES[signal?.code]
      ?? "The combined URL pattern was unusual, but no single lexical signal stood out.";
    return translate(template, requestedLanguage, signal?.values ?? {});
  }

  function translateNode(node) {
    if (!node?.nodeValue || !node.nodeValue.trim()) return;
    const trimmed = node.nodeValue.trim();
    const translated = translate(trimmed);
    if (translated === trimmed) return;
    const leading = node.nodeValue.match(/^\s*/)?.[0] ?? "";
    const trailing = node.nodeValue.match(/\s*$/)?.[0] ?? "";
    node.nodeValue = `${leading}${translated}${trailing}`;
  }

  function apply(rootNode = document) {
    if (language !== "ar") return;
    const rootElement = rootNode.nodeType === Node.ELEMENT_NODE ? rootNode : rootNode.documentElement;
    if (!rootElement) return;
    const walker = document.createTreeWalker(rootElement, NodeFilter.SHOW_TEXT, {
      acceptNode(node) {
        return ["SCRIPT", "STYLE", "TEXTAREA"].includes(node.parentElement?.tagName)
          ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT;
      },
    });
    while (walker.nextNode()) translateNode(walker.currentNode);
    rootElement.querySelectorAll?.("[placeholder]").forEach((element) => {
      element.placeholder = translate(element.placeholder);
    });
    rootElement.querySelectorAll?.("[aria-label]").forEach((element) => {
      element.setAttribute("aria-label", translate(element.getAttribute("aria-label")));
    });
    if (document.title) document.title = translate(document.title);
  }

  async function selectedLanguage() {
    const stored = await chrome.storage.local.get(STORAGE_KEY);
    if (["ar", "en"].includes(stored[STORAGE_KEY])) return stored[STORAGE_KEY];
    return String(chrome.i18n?.getUILanguage?.() ?? navigator.language).toLowerCase().startsWith("ar") ? "ar" : "en";
  }

  function updateSwitches() {
    document.querySelectorAll("[data-language-toggle]").forEach((button) => {
      button.textContent = language === "ar" ? "English" : "العربية";
      button.setAttribute("aria-label", language === "ar" ? "Switch to English" : "التبديل إلى العربية");
    });
  }

  async function initialize() {
    language = await selectedLanguage();
    document.documentElement.lang = language;
    document.documentElement.dir = language === "ar" ? "rtl" : "ltr";
    apply();
    updateSwitches();
    document.querySelectorAll("[data-language-toggle]").forEach((button) => {
      button.addEventListener("click", async () => {
        await chrome.storage.local.set({ [STORAGE_KEY]: language === "ar" ? "en" : "ar" });
        location.reload();
      });
    });
    const observer = new MutationObserver((mutations) => {
      if (language !== "ar") return;
      for (const mutation of mutations) {
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.TEXT_NODE) translateNode(node);
          else if (node.nodeType === Node.ELEMENT_NODE) apply(node);
        });
      }
    });
    observer.observe(document.documentElement, { childList: true, subtree: true });
  }

  return { STORAGE_KEY, ARABIC, SIGNAL_MESSAGES, translate, signalText, apply, initialize, language: () => language };
});
