#include "MainWindow.h"
#include <QFileDialog>
#include <QMessageBox>
#include <QApplication>
#include <QPainter>
#include <QPen>
#include <filesystem>
#include <algorithm>
#include <random>
#include <numeric>

namespace fs = std::filesystem;

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
    engine = new FacePCA(50);
    isModelTrained = false;

    face_cascade.load("D:\Alaa\Third_Year_SBME27\Second_Semester\Computer Vision\Task5\haarcascade_frontalface_default.xml");

    this->setStyleSheet("background-color: #121212; color: white; font-family: 'Segoe UI';");
    this->setWindowTitle("Final Project: Face Detection & Recognition");
    this->resize(1200, 700);

    setupLayout();
}

MainWindow::~MainWindow() {
    delete engine;
}

void MainWindow::setupLayout() {
    QWidget *central = new QWidget();
    QVBoxLayout *mainLayout = new QVBoxLayout(central);

    QLabel *header = new QLabel("CV PROJECT: DETECTION & RECOGNITION");
    header->setStyleSheet("font-size: 26px; font-weight: bold; color: #deff9a; padding: 10px;");
    header->setAlignment(Qt::AlignCenter);
    mainLayout->addWidget(header);

    tabs = new QTabWidget();
    tabs->setStyleSheet("QTabBar::tab { background: #333; color: white; padding: 15px; font-weight: bold; font-size: 14px;} "
                        "QTabBar::tab:selected { background: #deff9a; color: black; } "
                        "QTabWidget::pane { border: 2px solid #333; }");

    QWidget *tab1 = new QWidget();
    QWidget *tab2 = new QWidget();
    QWidget *tab3 = new QWidget();

    setupDetectionTab(tab1);
    setupRecognitionTab(tab2);
    setupPerformanceTab(tab3);

    tabs->addTab(tab1, "1. Face Detection");
    tabs->addTab(tab2, "2. Face Recognition (PCA)");
    tabs->addTab(tab3, "3. Performance & ROC");

    mainLayout->addWidget(tabs);
    setCentralWidget(central);
}

// ==========================================
// TAB 1: FACE DETECTION
// ==========================================
void MainWindow::setupDetectionTab(QWidget *tab) {
    QVBoxLayout *layout = new QVBoxLayout(tab);

    imgDetection = new QLabel("Upload an image to detect faces");
    imgDetection->setAlignment(Qt::AlignCenter);
    imgDetection->setStyleSheet("border: 2px dashed #555; background: #1a1a1a;");
    imgDetection->setMinimumHeight(400);
    layout->addWidget(imgDetection);

    QHBoxLayout *btnLayout = new QHBoxLayout();
    btnLoadDetImage = new QPushButton("Load Image");
    btnRunDetection = new QPushButton("Detect Faces");

    QString btnStyle = "QPushButton { background-color: #444; border: none; padding: 12px; font-weight: bold; border-radius: 5px; } QPushButton:hover { background-color: #555; }";
    btnLoadDetImage->setStyleSheet(btnStyle);
    btnRunDetection->setStyleSheet(btnStyle);

    btnLayout->addWidget(btnLoadDetImage);
    btnLayout->addWidget(btnRunDetection);
    layout->addLayout(btnLayout);

    connect(btnLoadDetImage, SIGNAL(clicked()), this, SLOT(handleLoadDetectionImage()));
    connect(btnRunDetection, SIGNAL(clicked()), this, SLOT(handleRunDetection()));
}

void MainWindow::handleLoadDetectionImage() {
    QString fileName = QFileDialog::getOpenFileName(this, "Select Image", "D:/Lyan/3rd year/2nd term/CV/task5/dataset", "Images (*.png *.jpg *.jpeg *.pgm)");
    if (fileName.isEmpty()) return;

    currentDetectionImage = cv::imread(fileName.toStdString());
    displayImageOnLabel(currentDetectionImage, imgDetection);
}

void MainWindow::handleRunDetection() {
    if (currentDetectionImage.empty()) return;

    cv::Mat gray, resultImage;
    currentDetectionImage.copyTo(resultImage);
    if (resultImage.channels() == 3) cv::cvtColor(resultImage, gray, cv::COLOR_BGR2GRAY);
    else resultImage.copyTo(gray);

    cv::equalizeHist(gray, gray);

    std::vector<cv::Rect> faces;
    face_cascade.detectMultiScale(gray, faces, 1.1, 7, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(50, 50));

    for (const auto& face : faces) {
        cv::rectangle(resultImage, face, cv::Scalar(0, 255, 0), 3);
    }

    displayImageOnLabel(resultImage, imgDetection);
}

// ==========================================
// TAB 2: FACE RECOGNITION (PCA)
// ==========================================
void MainWindow::setupRecognitionTab(QWidget *tab) {
    QVBoxLayout *layout = new QVBoxLayout(tab);

    QHBoxLayout *contentLayout = new QHBoxLayout();

    imgInput = new QLabel("Probe Image");
    imgInput->setFixedSize(300, 350);
    imgInput->setAlignment(Qt::AlignCenter);
    imgInput->setStyleSheet("border: 2px dashed #555; background: #1a1a1a;");

    imgResult = new QLabel("Matched Subject");
    imgResult->setFixedSize(300, 350);
    imgResult->setAlignment(Qt::AlignCenter);
    imgResult->setStyleSheet("border: 2px solid #deff9a; background: #1a1a1a;");

    contentLayout->addWidget(imgInput);
    contentLayout->addWidget(imgResult);
    layout->addLayout(contentLayout);

    QHBoxLayout *btnLayout = new QHBoxLayout();
    btnTrain = new QPushButton("1. Train PCA Dataset");
    btnRecognize = new QPushButton("2. Test Face Match");

    QString btnStyle = "QPushButton { background-color: #444; border: none; padding: 12px; font-weight: bold; border-radius: 5px; } QPushButton:hover { background-color: #555; border: 1px solid #deff9a; }";
    btnTrain->setStyleSheet(btnStyle);
    btnRecognize->setStyleSheet(btnStyle);

    btnLayout->addWidget(btnTrain);
    btnLayout->addWidget(btnRecognize);
    layout->addLayout(btnLayout);

    lblStatus = new QLabel("Recognition Status: Waiting for Training...");
    lblStatus->setAlignment(Qt::AlignCenter);
    lblStatus->setStyleSheet("color: #aaa; font-size: 15px; font-weight: bold; padding: 10px;");
    layout->addWidget(lblStatus);

    connect(btnTrain, SIGNAL(clicked()), this, SLOT(handleTrainPCA()));
    connect(btnRecognize, SIGNAL(clicked()), this, SLOT(handleRecognizeFace()));
}

void MainWindow::handleTrainPCA() {
    lblStatus->setText("Status: Loading Dataset & Training PCA... Please wait.");
    QApplication::processEvents();

    std::string dataset_path = "D:/Lyan/3rd year/2nd term/CV/task5/dataset";
    cv::Mat X_train;
    std::vector<int> y_train;

    for (int subject = 1; subject <= 10; ++subject) {
        std::string subject_path = dataset_path + "/s" + std::to_string(subject);
        if (!fs::exists(subject_path)) continue;

        for (int i = 1; i <= 7; ++i) {
            std::string img_path = subject_path + "/" + std::to_string(i) + ".pgm";
            if (!fs::exists(img_path)) continue;

            cv::Mat face_vector = engine->detectAndPreprocess(img_path, face_cascade);
            if (!face_vector.empty()) {
                X_train.push_back(face_vector);
                y_train.push_back(subject);
            }
        }
    }

    if (X_train.empty()) {
        lblStatus->setText("Error: Dataset not found or no faces detected!");
        return;
    }

    int num_samples = X_train.rows;
    std::vector<int> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    cv::Mat X_train_shuffled;
    std::vector<int> y_train_shuffled;

    for (int i = 0; i < num_samples; ++i) {
        X_train_shuffled.push_back(X_train.row(indices[i]));
        y_train_shuffled.push_back(y_train[indices[i]]);
    }

    engine->train(X_train_shuffled, y_train_shuffled);
    isModelTrained = true;

    lblStatus->setStyleSheet("color: #deff9a; font-size: 15px; font-weight: bold; padding: 10px;");
    lblStatus->setText("Status: Training Complete for 10 Subjects! Ready.");
}

void MainWindow::handleRecognizeFace() {
    if (!isModelTrained) {
        QMessageBox::warning(this, "Warning", "Please Train the PCA model first (Step 1)!");
        return;
    }

    QString fileName = QFileDialog::getOpenFileName(this, "Select Probe Image", "D:/Lyan/3rd year/2nd term/CV/task5/dataset", "Images (*.pgm *.jpg *.png)");
    if (fileName.isEmpty()) return;

    std::string img_path = fileName.toStdString();
    displayImageOnLabel(cv::imread(img_path, cv::IMREAD_GRAYSCALE), imgInput);

    lblStatus->setText("Analyzing Face...");
    QApplication::processEvents();

    cv::Mat face_vector = engine->detectAndPreprocess(img_path, face_cascade);
    if (face_vector.empty()) {
        lblStatus->setText("Status: No face detected!");
        lblStatus->setStyleSheet("color: red; font-size: 15px; font-weight: bold; padding: 10px;");
        return;
    }

    cv::Mat test_weight = engine->project(face_vector);
    double raw_distance;
    int predicted_label = engine->predict(test_weight.row(0), raw_distance);

    std::string matchPath = "D:/Lyan/3rd year/2nd term/CV/task5/dataset/s" + std::to_string(predicted_label) + "/1.pgm";
    displayImageOnLabel(cv::imread(matchPath, cv::IMREAD_GRAYSCALE), imgResult);

    // عرض الـ raw_distance الأصلية
    lblStatus->setText(QString("Status: Match Found! Subject S%1 (Dist: %2)").arg(predicted_label).arg(raw_distance, 0, 'f', 2));
    lblStatus->setStyleSheet("color: #deff9a; font-size: 15px; font-weight: bold; padding: 10px;");
}

// ==========================================
// TAB 3: PERFORMANCE & SIDE-BY-SIDE ROC CURVES
// ==========================================
void MainWindow::setupPerformanceTab(QWidget *tab) {
    QVBoxLayout *layout = new QVBoxLayout(tab);

    lblAccuracy = new QLabel("System Accuracy: N/A");
    lblAccuracy->setAlignment(Qt::AlignCenter);
    lblAccuracy->setStyleSheet("font-size: 28px; font-weight: bold; color: #deff9a;");
    layout->addWidget(lblAccuracy);

    QHBoxLayout *graphsLayout = new QHBoxLayout();

    QVBoxLayout *g1Layout = new QVBoxLayout();
    QLabel *l1 = new QLabel("Overall System ROC (Genuine vs Impostor)");
    l1->setAlignment(Qt::AlignCenter);
    l1->setStyleSheet("font-weight: bold; color: #fff; font-size: 14px;");

    imgGlobalRoc = new QLabel("Global Curve Area");
    imgGlobalRoc->setAlignment(Qt::AlignCenter);
    imgGlobalRoc->setStyleSheet("border: 1px solid #555; background: #1a1a1a;");
    imgGlobalRoc->setFixedSize(500, 400);

    g1Layout->addWidget(l1);
    g1Layout->addWidget(imgGlobalRoc);

    QVBoxLayout *g2Layout = new QVBoxLayout();
    QLabel *l2 = new QLabel("Multi-class ROC (One-vs-Rest)");
    l2->setAlignment(Qt::AlignCenter);
    l2->setStyleSheet("font-weight: bold; color: #fff; font-size: 14px;");

    imgMultiRoc = new QLabel("Multi-class Area");
    imgMultiRoc->setAlignment(Qt::AlignCenter);
    imgMultiRoc->setStyleSheet("border: 1px solid #555; background: #1a1a1a;");
    imgMultiRoc->setFixedSize(500, 400);

    g2Layout->addWidget(l2);
    g2Layout->addWidget(imgMultiRoc);

    graphsLayout->addLayout(g1Layout);
    graphsLayout->addLayout(g2Layout);
    layout->addLayout(graphsLayout);

    btnEvaluate = new QPushButton("Select Test Dataset Folder & Evaluate");
    QString btnStyle = "QPushButton { background-color: #444; border: none; padding: 15px; font-weight: bold; border-radius: 5px; } QPushButton:hover { background-color: #555; border: 1px solid #00ffff; }";
    btnEvaluate->setStyleSheet(btnStyle);
    layout->addWidget(btnEvaluate);

    connect(btnEvaluate, SIGNAL(clicked()), this, SLOT(handleEvaluate()));
}

void MainWindow::handleEvaluate() {
    if (!isModelTrained) {
        QMessageBox::warning(this, "Error", "Please Train the PCA model first (in Tab 2) before evaluating!");
        return;
    }

    QString dir = QFileDialog::getExistingDirectory(this, "Select Testing Dataset Folder", "D:/Lyan/3rd year/2nd term/CV/task5/dataset");
    if (dir.isEmpty()) return;

    lblAccuracy->setText("Evaluating... Please Wait");
    QApplication::processEvents();

    struct TestSample {
        int true_label;
        int pred_label;
        double distance;
    };
    std::vector<TestSample> all_tests;

    int correct = 0, total = 0;
    double max_dist = 0;

    for (int subject = 1; subject <= 10; ++subject) {
        std::string path = dir.toStdString() + "/s" + std::to_string(subject);
        if (!fs::exists(path)) continue;

        for (const auto& entry : fs::directory_iterator(path)) {
            std::string img_path = entry.path().string();
            if (img_path.find(".pgm") == std::string::npos && img_path.find(".jpg") == std::string::npos && img_path.find(".png") == std::string::npos) continue;

            cv::Mat face_vector = engine->detectAndPreprocess(img_path, face_cascade);
            if (!face_vector.empty()) {
                double raw_dist;
                int pred = engine->predict(engine->project(face_vector).row(0), raw_dist);

                // إضافة الـ raw_dist مباشرة بدون قسمة
                all_tests.push_back({subject, pred, raw_dist});
                if (pred == subject) correct++;
                total++;
                if (raw_dist > max_dist) max_dist = raw_dist;
            }
        }
    }

    if (total == 0) {
        lblAccuracy->setText("Error: No valid test images found!");
        return;
    }

    int steps = 100;

    std::vector<std::pair<double, double>> global_points;
    for (int i = 0; i <= steps; ++i) {
        double threshold = (max_dist / steps) * i;
        int tp = 0, fp = 0, genuine_count = 0, impostor_count = 0;

        for (const auto& ts : all_tests) {
            if (ts.true_label == ts.pred_label) genuine_count++;
            else impostor_count++;

            if (ts.distance <= threshold) {
                if (ts.true_label == ts.pred_label) tp++;
                else fp++;
            }
        }

        double tpr = (genuine_count > 0) ? (double)tp / genuine_count : 0;
        double fpr = (impostor_count > 0) ? (double)fp / impostor_count : 0;
        global_points.push_back({fpr, tpr});
    }
    std::sort(global_points.begin(), global_points.end());
    drawGlobalROC(global_points);

    std::vector<std::vector<std::pair<double, double>>> multi_curves;
    for (int class_id = 1; class_id <= 10; ++class_id) {
        std::vector<std::pair<double, double>> curve;
        for (int i = 0; i <= steps; ++i) {
            double threshold = (max_dist / steps) * i;
            int tp = 0, fp = 0, pos_total = 0, neg_total = 0;

            for (const auto& ts : all_tests) {
                if (ts.true_label == class_id) {
                    pos_total++;
                    if (ts.pred_label == class_id && ts.distance <= threshold) tp++;
                } else {
                    neg_total++;
                    if (ts.pred_label == class_id && ts.distance <= threshold) fp++;
                }
            }
            double tpr = (pos_total > 0) ? (double)tp / pos_total : 0;
            double fpr = (neg_total > 0) ? (double)fp / neg_total : 0;
            curve.push_back({fpr, tpr});
        }
        std::sort(curve.begin(), curve.end());
        multi_curves.push_back(curve);
    }
    drawMultiClassROC(multi_curves);

    double acc = ((double)correct / total) * 100.0;
    lblAccuracy->setText(QString("System Accuracy: %1%").arg(QString::number(acc, 'f', 2)));
}

void MainWindow::drawGlobalROC(const std::vector<std::pair<double, double>>& points) {
    QPixmap pix(imgGlobalRoc->size());
    pix.fill(QColor("#1a1a1a"));
    QPainter p(&pix);
    p.setRenderHint(QPainter::Antialiasing);
    int pad = 40, w = pix.width() - 2*pad, h = pix.height() - 2*pad;

    p.setPen(QPen(Qt::white, 2));
    p.drawLine(pad, pix.height()-pad, pix.width()-pad, pix.height()-pad);
    p.drawLine(pad, pad, pad, pix.height()-pad);
    p.drawText(pix.width()/2 - 10, pix.height() - 10, "FPR");

    p.translate(15, pix.height()/2 + 10);
    p.rotate(-90);
    p.drawText(0, 0, "TPR");
    p.rotate(90);
    p.translate(-15, -(pix.height()/2 + 10));

    p.setPen(QPen(Qt::gray, 1, Qt::DashLine));
    p.drawLine(pad, pix.height()-pad, pix.width()-pad, pad);

    p.setPen(QPen(QColor("#00ffff"), 3));
    for (size_t i = 1; i < points.size(); ++i) {
        p.drawLine(pad + points[i-1].first * w, (pix.height()-pad) - points[i-1].second * h,
                   pad + points[i].first * w, (pix.height()-pad) - points[i].second * h);
    }
    imgGlobalRoc->setPixmap(pix);
}

void MainWindow::drawMultiClassROC(const std::vector<std::vector<std::pair<double, double>>>& curves) {
    QPixmap pix(imgMultiRoc->size());
    pix.fill(QColor("#1a1a1a"));
    QPainter p(&pix);
    p.setRenderHint(QPainter::Antialiasing);
    int pad = 40, w = pix.width() - 2*pad, h = pix.height() - 2*pad;

    p.setPen(QPen(Qt::white, 2));
    p.drawLine(pad, pix.height()-pad, pix.width()-pad, pix.height()-pad);
    p.drawLine(pad, pad, pad, pix.height()-pad);
    p.drawText(pix.width()/2 - 10, pix.height() - 10, "FPR");

    p.translate(15, pix.height()/2 + 10);
    p.rotate(-90);
    p.drawText(0, 0, "TPR");
    p.rotate(90);
    p.translate(-15, -(pix.height()/2 + 10));

    QColor colors[] = {
        Qt::red, Qt::green, QColor("#1f77b4"), Qt::yellow,
        Qt::magenta, Qt::cyan, QColor("#ff7f0e"),
        QColor("#8c564b"), QColor("#bcbd22"), Qt::white
    };

    for (int c = 0; c < 10 && c < curves.size(); ++c) {
        p.setPen(QPen(colors[c], 2));
        for (size_t i = 1; i < curves[c].size(); ++i) {
            p.drawLine(pad + curves[c][i-1].first * w, (pix.height()-pad) - curves[c][i-1].second * h,
                       pad + curves[c][i].first * w, (pix.height()-pad) - curves[c][i].second * h);
        }
    }
    imgMultiRoc->setPixmap(pix);
}

void MainWindow::displayImageOnLabel(const cv::Mat& img, QLabel* label) {
    if(img.empty()) return;

    cv::Mat rgb;
    if (img.channels() == 1) cv::cvtColor(img, rgb, cv::COLOR_GRAY2RGB);
    else cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

    QImage qimg((const unsigned char*)(rgb.data), rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888);
    label->setPixmap(QPixmap::fromImage(qimg).scaled(label->size(), Qt::KeepAspectRatio, Qt::SmoothTransformation));
}
