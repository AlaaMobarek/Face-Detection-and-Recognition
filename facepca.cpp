#include "facepca.h"

FacePCA::FacePCA(int components) : num_components(components) {}

cv::Mat FacePCA::detectAndPreprocess(const std::string& imagePath, cv::CascadeClassifier& cascade) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) return cv::Mat();

    std::vector<cv::Rect> faces;
    cascade.detectMultiScale(img, faces, 1.1, 4);

    cv::Mat face_roi;
    if (!faces.empty()) {
        face_roi = img(faces[0]); // قص الوش
    } else {
        face_roi = img; // لو ملقاش وش، ياخد الصورة كلها كاحتياطي
    }

    cv::Mat resized_face;
    cv::resize(face_roi, resized_face, cv::Size(50, 50));

    // تحويل لصف واحد (Flatten) وتحويل الأرقام لـ Float عشان الحسابات
    cv::Mat flattened = resized_face.reshape(1, 1);
    cv::Mat float_face;
    flattened.convertTo(float_face, CV_32FC1);

    return float_face;
}

void FacePCA::train(const cv::Mat& training_data, const std::vector<int>& labels) {
    training_labels = labels;
    int num_images = training_data.rows;

    // 1. حساب المتوسط (Mean Face)
    cv::reduce(training_data, mean_face, 0, cv::REDUCE_AVG);

    // 2. طرح المتوسط من كل الصور (A = X - mean)
    cv::Mat A = cv::Mat::zeros(training_data.size(), training_data.type());
    for (int i = 0; i < num_images; i++) {
        A.row(i) = training_data.row(i) - mean_face;
    }

    // 3. حساب Covariance Matrix بطريقة مختصرة (L = A * A^T)
    cv::Mat L = A * A.t();

    // 4. استخراج Eigenvalues و Eigenvectors
    cv::Mat evalues, evectors_L;
    cv::eigen(L, evalues, evectors_L);

    // 5. حساب الـ Eigenfaces الحقيقية (V = A^T * evectors_L^T)
    cv::Mat actual_eigenfaces = A.t() * evectors_L.t();

    // 6. Normalization وتخزين أفضل K Components
    eigenfaces = cv::Mat::zeros(actual_eigenfaces.rows, num_components, CV_32FC1);
    for (int i = 0; i < num_components; i++) {
        cv::Mat col = actual_eigenfaces.col(i);
        cv::normalize(col, col);
        col.copyTo(eigenfaces.col(i));
    }

    // 7. حساب أوزان صور التدريب
    training_weights = project(training_data);
}

cv::Mat FacePCA::project(const cv::Mat& faces) {
    cv::Mat A = cv::Mat::zeros(faces.size(), faces.type());
    for (int i = 0; i < faces.rows; i++) {
        A.row(i) = faces.row(i) - mean_face;
    }
    return A * eigenfaces; // W = A * U
}

int FacePCA::predict(const cv::Mat& test_face_weight, double& out_distance) {
    double min_dist = DBL_MAX;
    int best_label = -1;

    // حساب المسافة الإقليدية
    for (int i = 0; i < training_weights.rows; i++) {
        double dist = cv::norm(training_weights.row(i), test_face_weight, cv::NORM_L2);
        if (dist < min_dist) {
            min_dist = dist;
            best_label = training_labels[i];
        }
    }
    out_distance = min_dist;
    return best_label;
}
