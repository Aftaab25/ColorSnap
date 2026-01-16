#include "utils/csv_file_reader.h"
#include "utils/rgb2hex.h"
#include <fmt/core.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

struct ClickData
{
    bool clicked = false;
    cv::Point position;
    RGB color;
    cv::Mat *img = nullptr;
};

struct ColorEntry
{
    std::string color;
    std::string name;
    std::string hex;
    int r, g, b;
};

std::vector<ColorEntry> load_colors(const std::string filename)
{
    auto raw = read_csv(filename);
    std::vector<ColorEntry> dataset;

    for (const auto &row : raw)
    {
        if (row.size() < 6)
            continue;

        dataset.push_back(
            {row[0], row[1], row[2], std::stoi(row[3]), std::stoi(row[4]), std::stoi(row[5])});
    }

    return dataset;
}

std::pair<std::string, int> getColor(const RGB &rgb, std::vector<ColorEntry> &dataset)
{
    int minDist = INT_MAX;
    std::string found;
    for (const auto &entry : dataset)
    {
        int distance = abs(rgb.r - entry.r) + abs(rgb.g - entry.g) + abs(rgb.b - entry.b);
        if (distance <= minDist)
        {
            minDist = distance;
            found = entry.color;
        }
    }

    return std::make_pair(found, minDist);
}

void drawRoundedRectangle(cv::Mat &img, cv::Rect rect, cv::Scalar color, int radius)
{
    int x = rect.x;
    int y = rect.y;
    int w = rect.width;
    int h = rect.height;

    // Draw the main body (cross shape)
    cv::rectangle(img, cv::Point(x + radius, y), cv::Point(x + w - radius, y + h), color, -1,
                  cv::LINE_AA);
    cv::rectangle(img, cv::Point(x, y + radius), cv::Point(x + w, y + h - radius), color, -1,
                  cv::LINE_AA);

    // Draw the four corners
    cv::circle(img, cv::Point(x + radius, y + radius), radius, color, -1, cv::LINE_AA);
    cv::circle(img, cv::Point(x + w - radius, y + radius), radius, color, -1, cv::LINE_AA);
    cv::circle(img, cv::Point(x + radius, y + h - radius), radius, color, -1, cv::LINE_AA);
    cv::circle(img, cv::Point(x + w - radius, y + h - radius), radius, color, -1, cv::LINE_AA);
}

void drawLabel(cv::Mat &img, const std::string &text, cv::Point pos, const RGB &bgColorRGB)
{
    int fontFace = cv::FONT_HERSHEY_DUPLEX;
    double fontScale = 0.7;
    int thickness = 1;
    int baseline = 0;

    cv::Size textSize = cv::getTextSize(text, fontFace, fontScale, thickness, &baseline);

    int padding = 10;
    int rectWidth = textSize.width + 2 * padding;
    int rectHeight = textSize.height + 2 * padding;

    // Dynamic positioning: Float above cursor by default
    int offset = 20;
    cv::Point topLeft;
    topLeft.x = pos.x - rectWidth / 2;
    topLeft.y = pos.y - rectHeight - offset;

    // Clamp to image bounds
    if (topLeft.x < 10)
        topLeft.x = 10;
    if (topLeft.x + rectWidth > img.cols - 10)
        topLeft.x = img.cols - rectWidth - 10;
    if (topLeft.y < 10)
        topLeft.y = pos.y + offset; // Flip to below if hits top
    if (topLeft.y + rectHeight > img.rows - 10)
        topLeft.y = img.rows - rectHeight - 10;

    cv::Rect bgRect(topLeft.x, topLeft.y, rectWidth, rectHeight);

    // Shadow
    int shadowOffset = 5;
    cv::Rect shadowRect = bgRect;
    shadowRect.x += shadowOffset;
    shadowRect.y += shadowOffset;
    drawRoundedRectangle(img, shadowRect, cv::Scalar(50, 50, 50), 10);

    // Background
    cv::Scalar bgScalar(bgColorRGB.b, bgColorRGB.g, bgColorRGB.r); // BGR
    drawRoundedRectangle(img, bgRect, bgScalar, 10);

    // Text Color based on Luminance
    double luminance = 0.299 * bgColorRGB.r + 0.587 * bgColorRGB.g + 0.114 * bgColorRGB.b;
    cv::Scalar textColor = (luminance > 128) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

    // Centered Text Position
    cv::Point textOrg;
    textOrg.x = topLeft.x + padding;
    textOrg.y = topLeft.y + (rectHeight + textSize.height) / 2 - 2;

    cv::putText(img, text, textOrg, fontFace, fontScale, textColor, thickness, cv::LINE_AA);
}

void copyToClipboard(const std::string &text)
{
    FILE *pipe = popen("xsel --clipboard --input", "w");
    if (!pipe)
        return;
    fwrite(text.c_str(), 1, text.size(), pipe);
    pclose(pipe);
    std::cout << "Copied to clipboard:\n" << text << std::endl;
}

int main(int argc, char *argv[])
{
    std::string image_path; // = "assets/images/colorpic.jpg";

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];

        if ((arg == "-i" || arg == "--image") && i + 1 < argc)
        {
            image_path = argv[i + 1];
            i++; // skip next since it's the value
        }
    }

    if (image_path.empty())
    {
        std::cerr << "Usage: " << argv[0] << " -i <image_path>\n";
        return 1;
    }

    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);

    if (img.empty())
    {
        std::cerr << "Error: No image found.\n";
        return -1;
    }

    auto dataset = load_colors("assets/data/colors.csv");

    cv::Mat resized_img;
    cv::Size new_size(1000, 800);

    resize(img, resized_img, new_size, 0, 0, cv::INTER_LINEAR);
    cv::Mat clean_img = resized_img.clone();

    ClickData data;
    data.img = &resized_img;

    RGB last_color = {0, 0, 0};
    bool has_selection = false;

    cv::namedWindow("image");

    cv::setMouseCallback(
        "image",
        [](int event, int x, int y, int flags, void *userdata)
        {
            if (event == cv::EVENT_LBUTTONDBLCLK)
            {
                auto *d = reinterpret_cast<ClickData *>(userdata);
                d->clicked = true;
                d->position = cv::Point(x, y);

                cv::Vec3b pixel = d->img->at<cv::Vec3b>(y, x); // BGR order
                std::cout << pixel << std::endl;
                d->color.b = static_cast<int>(pixel[0]);
                d->color.g = static_cast<int>(pixel[1]);
                d->color.r = static_cast<int>(pixel[2]);
            }
        },
        &data);

    for (;;)
    {
        imshow("image", resized_img);

        if (data.clicked)
        {
            // Reset image to clean state to remove previous labels
            clean_img.copyTo(resized_img);

            last_color = data.color;
            has_selection = true;

            std::pair<std::string, int> result = getColor(data.color, dataset);
            std::string colorFound = result.first;
            double accuracy = ((double)(765 - result.second) / 765) * 100;
            std::string accuracy_value = fmt::format("{:.2f}%", accuracy);

            std::string text = colorFound + " R=" + std::to_string(data.color.r) +
                               " G=" + std::to_string(data.color.g) +
                               " B=" + std::to_string(data.color.b) + " " + rgb2hex(data.color) +
                               " " + accuracy_value + "%" + " | [c] Copy";

            drawLabel(resized_img, text, data.position, data.color);

            data.clicked = false;
        }

        int key = cv::waitKey(20);
        if (key == 27 || key == 'q')
        {
            break;
        }
        else if (key == 'c' && has_selection)
        {
            std::pair<std::string, int> result = getColor(last_color, dataset);
            std::string colorFound = result.first;
            double accuracy = ((double)(765 - result.second) / 765) * 100;
            std::string accuracy_value = fmt::format("{:.2f}%", accuracy);

            std::string clipboardText = fmt::format("Color: {}\nRGB: ({}, {}, {})\nHEX: {}\nAccuracy: {}",
                                                    colorFound, last_color.r, last_color.g, last_color.b,
                                                    rgb2hex(last_color), accuracy_value);
            copyToClipboard(clipboardText);
        }
    }

    cv::destroyAllWindows();

    return 0;
}