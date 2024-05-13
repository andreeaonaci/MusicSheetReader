#include "stdafx.h"
#include <vector>
#include <stack>
#include <random>
#include <iostream>
#include <fstream>
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace std;

enum NoteName { C, D, E, F, G, A, B };

enum Duration { quarter, eighth }; // Define quarter and eighth notes

struct Note {
    NoteName name;
    int octave;
    Duration duration;
    int position; // Position relative to the staff line
};

struct Line {
    int y;
};

struct Staff {
    vector<Line> lines;
};

vector<Staff> detectStaffs(Mat image, bool print) {

    vector<int> horizontalProjection(image.rows, 0);
    for (int i = 0; i < image.rows; ++i) {
        int blackPixelCount = 0;
        for (int j = 0; j < image.cols; ++j) {
            if (image.at<uchar>(i, j) == 0) {
                blackPixelCount++;
            }
        }
        horizontalProjection[i] = blackPixelCount;
    }

    cout << image.rows << " " << image.cols << endl;

    std::vector<int> candidateLines; //lines that are candidates for staff lines

    for (int i = 0; i < horizontalProjection.size(); ++i) {
        if (horizontalProjection[i] > image.cols / 2) {
            cout << "Candidate line at y = " << i << endl;  
			candidateLines.push_back(i);
        }
	}

    vector<Staff> staffs;
    Staff staff;

    staff.lines.push_back({ candidateLines[0] });

    for (int i = 1; i < candidateLines.size(); ++i) {
        Line line_candidate;
        line_candidate.y = candidateLines[i];
        if (candidateLines[i] - candidateLines[i - 1] != 1)
            staff.lines.push_back({ line_candidate });
        
        int j = i + 1;
        while (j < candidateLines.size() && candidateLines[j] - candidateLines[j - 1] == 1) {
            j++;
        }

        if (staff.lines.size() == 5) {
			staffs.push_back(staff);
			staff.lines.clear();
		}
    }

    if (print) {
        for (int i = 0; i < staffs.size(); ++i) {
            cout << "Staff " << i << " lines: ";
            for (int j = 0; j < staffs[i].lines.size(); ++j) {
                cout << staffs[i].lines[j].y << " ";
            }
            cout << endl;
        }

        Mat result = image.clone(); // Make a copy of the input image
        for (int i = 0; i < staffs.size(); ++i) {
            for (int j = 0; j < staffs[i].lines.size(); ++j) {
                line(result, Point(0, staffs[i].lines[j].y), Point(result.cols, staffs[i].lines[j].y), Scalar(0, 255, 0), 2);
            }
        }
        imshow("Detected Staffs", result);
        waitKey(0);
    }


    return staffs;
}


NoteName calculateNoteName(int positionWithinStaffLine) {
    // Assuming 7 unique notes per octave
    // Mapping the position within the staff line to the note name
    switch (positionWithinStaffLine % 7) {
    case 0: return C;
    case 1: return D;
    case 2: return E;
    case 3: return F;
    case 4: return G;
    case 5: return A;
    case 6: return B;
    default: return C; // Default to C if position is out of range
    }
}

// Function to detect notes relative to staff lines
vector<Note> detectNotes(const Mat& binaryImage, const vector<Rect>& staffLineRegions) {
    vector<Note> notes;

    for (const Rect& staffLineRegion : staffLineRegions) {
        Mat subRegion = binaryImage(staffLineRegion);
        int staffLinePosition = staffLineRegion.y + staffLineRegion.height / 2;

        // Calculate the position within the staff line relative to the top of the region
        for (int i = 0; i < subRegion.rows; ++i) {
            for (int j = 0; j < subRegion.cols; ++j) {
                if (subRegion.at<uchar>(i, j) > 0) {
                    int positionWithinStaffLine = i; // Position relative to the top of the staff line
                    NoteName noteName = calculateNoteName(positionWithinStaffLine);
                    // Create and store the note
                    Note note;
                    note.name = noteName;
                    notes.push_back(note);
                }
            }
        }
    }

    return notes;
}

bool isInside(Mat img, int i, int j) {
    if ((i < img.rows) && (j < img.cols) && (i >= 0) && (j >= 0))
        return true;
    else
        return false;

}

Mat_<uchar> dilationLab7(Mat_<uchar> img, Mat_<uchar> strel) {
    Mat_<uchar> dst(img.rows, img.cols, 255);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0) {
                for (int u = 0; u < strel.rows; u++) {
                    for (int v = 0; v < strel.cols; v++) {
                        if (!strel(u, v)) {
                            //vecinul care este sub elementul (u,v) din strel
                            int i2 = i + u - strel.rows / 2;
                            int j2 = j + v - strel.cols / 2;
                            if (isInside(dst, i2, j2))
                                dst(i2, j2) = 0;
                        }
                    }
                }
            }
        }
    }
    return dst;
}

Mat_<uchar> erosionLab7(Mat_<uchar> img, Mat_<uchar> strel) {
    Mat_<uchar> dst(img.rows, img.cols, 255);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0) {
                int allblack = 1;
                for (int u = 0; u < strel.rows; u++) {
                    for (int v = 0; v < strel.cols; v++) {
                        if (!strel(u, v)) {
                            //vecinul care este sub elementul (u,v) din strel
                            int i2 = i + u - strel.rows / 2;
                            int j2 = j + v - strel.cols / 2;
                            if (isInside(dst, i2, j2) && img(i2, j2) == 255)
                                allblack = 0;
                        }
                    }
                }
                if (allblack)
                    dst(i, j) = 0;
            }

        }
    }
    return dst;
}

Mat_<uchar> openingLab7(Mat_<uchar> img, Mat_<uchar> strel) {
    return dilationLab7(erosionLab7(img, strel), strel);
}

Mat_<uchar> closingLab7(Mat_<uchar> img, Mat_<uchar> strel) {
    return erosionLab7(dilationLab7(img, strel), strel);
}


Mat invertColors(Mat img) {
    Mat invertedImg;
    bitwise_not(img, invertedImg);
    return invertedImg;
}

uchar label = 0;
int newLabel = 0;

Mat connectedLabeling(Mat src) {
    label = 0;
    newLabel = 0;
    Mat labels = Mat::zeros(src.size(), CV_32SC1);
    Mat colorImg(src.rows, src.cols, CV_8UC3, Scalar(255, 255, 255));
    std::vector<std::vector<int>> edges(1);
    int di[8] = { -1,-1,-1, 0, 0, 1, 1, 1 };
    int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
                std::vector<int> neighbours;
                for (int k = 0; k < 4; k++) {
                    int ni = i + di[k];
                    int nj = j + dj[k];
                    if (ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols && labels.at<int>(ni, nj) > 0) {
                        neighbours.push_back(labels.at<int>(ni, nj));
                    }
                }
                if (neighbours.size() == 0) {
                    label++;
                    labels.at<int>(i, j) = label;
                    edges.push_back(std::vector<int>());
                }
                else {
                    int min = neighbours[0];
                    for (int k = 1; k < neighbours.size(); k++) {
                        if (neighbours[k] < min) {
                            min = neighbours[k];
                        }
                    }
                    labels.at<int>(i, j) = min;
                    for (int k = 0; k < neighbours.size(); k++) {
                        if (neighbours[k] != min)
                        {
                            edges.at(min).push_back(neighbours[k]);
                            edges.at(neighbours[k]).push_back(min);
                        }

                    }
                }
            }
        }
    }

    Mat newLabels(label + 1, 1, CV_32SC1, Scalar(0));
    for (int i = 1; i <= label; i++) {
        if (newLabels.at<int>(i) == 0) {
            newLabel++;
            std::queue<int> Q;
            Q.push(i);
            newLabels.at<int>(i) = newLabel;
            while (!Q.empty()) {
                int q = Q.front();
                Q.pop();
                for (int k = 0; k < edges[q].size(); k++) {
                    if (newLabels.at<int>(edges[q][k]) == 0) {
                        newLabels.at<int>(edges[q][k]) = newLabel;
                        Q.push(edges[q][k]);
                    }
                }
            }
        }
    }

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            if (labels.at<int>(i, j) != 0) {
                labels.at<int>(i, j) = newLabels.at<int>(labels.at<int>(i, j));
            }
        }
    }

    std::default_random_engine gen;
    std::uniform_int_distribution<int> d(0, 255);
    for (int k = 2; k <= newLabel; k++) {
        Vec3b color = Vec3b(rand() % 256, rand() % 256, rand() % 256);
        for (int i = 1; i < src.rows - 1; i++) {
            for (int j = 1; j < src.cols - 1; j++) {
                if (labels.at<int>(i, j) == k)
                    colorImg.at<Vec3b>(i, j) = color;
            }
        }
    }

    imshow("color", colorImg);
    waitKey(0);

    return labels;
}

Point calculateCenterOfMassFromPoint(const Mat binary, Point point, int label) {
    Mat src = binary.clone();

    if (point.x < 0 || point.y < 0 || point.x >= src.cols || point.y >= src.rows) {
        return Point(-1, -1); 
    }

    float totalArea = 0;
    float sumX = 0, sumY = 0;
    bool flag = 0;

    queue<Point> q;
    q.push(point);

    int di[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    int dj[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };

    while (!q.empty()) {
        Point current = q.front();
        q.pop();
        totalArea++;
        sumX += current.x;
        sumY += current.y;

        for (int d = 0; d < 8; ++d) {
            Point next(current.x + di[d], current.y + dj[d]);
            if (next.x >= 0 && next.x < src.cols && next.y >= 0 && next.y < src.rows && src.at<int>(next) == label) {
                src.at<int>(next) = 255;
                q.push(next);
                if (next.x < point.x - 5) {
                    flag = 1;
				}
            }
        }
    }
    
    int centerX = static_cast<int>(sumX / totalArea);
    int centerY = static_cast<int>(sumY / totalArea); 

    return Point(centerX, centerY);
}



vector<int> calculateVerticalProjection(const Mat& binaryImage) {
    vector<int> verticalProjection(binaryImage.cols, 0);
    for (int j = 0; j < binaryImage.cols; ++j) {
        for (int i = 0; i < binaryImage.rows; ++i) {
            if (binaryImage.at<uchar>(i, j) == 0) {
                verticalProjection[j]++;
            }
        }
    }
    return verticalProjection;
}

void removeGreaterObjects(Mat labels, int threshold) {
    for (int label = 1; label <= newLabel; ++label) {
        int area = 0;

        for (int i = 0; i < labels.rows; ++i) {
            for (int j = 0; j < labels.cols; ++j) {
                if (labels.at<int>(i, j) == label) {
                    area++;
                }
            }
        }

        cout << "Area for label " << label << ": " << area << endl;

        if (area < 23 || area > 50) {
            for (int i = 0; i < labels.rows; ++i) {
                for (int j = 0; j < labels.cols; ++j) {
                    if (labels.at<int>(i, j) == label) {
                        labels.at<int>(i, j) = 0;
                    }
                }
            }
        }
    }
}


void sortLabelsByCenterOfMass(const Mat& labels, const vector<Staff>& staffs, vector<pair<Point, int>>& sortedLabels, Mat image) {
    sortedLabels.clear(); 
    cout << "Sunt aici";
    Mat_<Vec3b> srcColor = image.clone();
    vector<int> visitedLabels(newLabel + 1, 0);

    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            int label = labels.at<int>(i, j);
            if (label != 0 && visitedLabels[label] == 0) {
                Point center = calculateCenterOfMassFromPoint(labels, Point(j, i), label);
                sortedLabels.push_back({ center, label });
                visitedLabels[label] = 1;
            }
        }
    }

    auto comparator = [](const pair<Point, int>& a, const pair<Point, int>& b) {
        if (a.first.y - 45 < b.first.y && a.first.y + 45 > b.first.y) {
            return a.first.x < b.first.x;
        }
        else {
            return a.first.y < b.first.y;
        }
        };

    sort(sortedLabels.begin(), sortedLabels.end(), comparator);

    vector<int> extractedLabels;
    for (const auto& pair : sortedLabels) {
        extractedLabels.push_back(pair.second);
    }

    for (int i = 0; i < extractedLabels.size(); i++) {
        Point center = sortedLabels[i].first;
        cout << "Center of mass for label " << extractedLabels[i] << ": " << center << endl;
        circle(srcColor, center, 3, Scalar(0, 0, 255), FILLED);
        imshow("Detected Notes", srcColor);
        waitKey(0);
    }
}

vector<Note> identifyNotes(const Mat labels, const vector<Staff>& staffs, Mat image) {
    vector<Note> notes;

    vector<pair<Point, int>> sortedLabels;

    removeGreaterObjects(labels, 100);

    sortLabelsByCenterOfMass(labels, staffs, sortedLabels, image);

    int staffNumber = 0;
    for (int i = 0; i < sortedLabels.size(); i++) {
        Point center = sortedLabels[i].first;
        int offset = 3;
        int tolerance = 5;

        if (center.y < staffs[staffNumber].lines[0].y - tolerance || center.y > staffs[staffNumber].lines[4].y + 2 * tolerance) {
            staffNumber++;
		}

        if (staffNumber == staffs.size()) {
            break;
        }

        if (center.y < staffs[staffNumber].lines[0].y - offset) {
            Note note;
            note.name = G;
            note.octave = 5;
            note.duration = quarter;
            note.position = 0;
            notes.push_back(note);
		}
        else
        if (center.y < staffs[staffNumber].lines[0].y + offset) {
			Note note;
			note.name = F;
			note.octave = 5;
			note.duration = quarter;
			note.position = 0;
			notes.push_back(note);
		}
        else 
        if (center.y < staffs[staffNumber].lines[1].y - offset) {
            Note note;
            note.name = E;
            note.octave = 5;
            note.duration = quarter;
            note.position = 0;
            notes.push_back(note);
        }
        else
        if (center.y < staffs[staffNumber].lines[1].y + offset) {
			Note note;
			note.name = D;
			note.octave = 5;
			note.duration = quarter;
			note.position = 0;
			notes.push_back(note);
		}
        else
        if (center.y < staffs[staffNumber].lines[2].y - offset) {
			Note note;
			note.name = C;
			note.octave = 5;
			note.duration = quarter;
			note.position = 0;
			notes.push_back(note);
		}
        else
        if (center.y < staffs[staffNumber].lines[2].y + offset) {
			Note note;
			note.name = B;
			note.octave = 4;
			note.duration = quarter;
			note.position = 0;
			notes.push_back(note);
		}
        else
        if (center.y < staffs[staffNumber].lines[3].y - offset) {
			Note note;
			note.name = A;
			note.octave = 4;
			note.duration = quarter;
			note.position = 0;
			notes.push_back(note);
		}
        else
        if (center.y < staffs[staffNumber].lines[3].y + offset) {
			Note note;
			note.name = G;
			note.octave = 4;
			note.duration = quarter;
			note.position = 0;
			notes.push_back(note);
		}
        else
        if (center.y < staffs[staffNumber].lines[4].y - offset) {
			Note note;
			note.name = F;
			note.octave = 4;
			note.duration = quarter;
			note.position = 0;
			notes.push_back(note);
		}
        else
        if (center.y < staffs[staffNumber].lines[4].y + offset) {
			Note note;
			note.name = E;
			note.octave = 4;
			note.duration = quarter;
			note.position = 0;
			notes.push_back(note);
		}
        else
            if (center.y > staffs[staffNumber].lines[4].y + offset) {
                cout << "Note center " << center.x << endl;
				Note note;
				note.name = D;
				note.octave = 4;
                note.duration = quarter;
                note.position = 0;
                notes.push_back(note);
            }
        else
            if (center.y > staffs[staffNumber].lines[4].y + 20 * offset) {
                Note note;
                note.name = C;
                note.octave = 4;
                note.duration = quarter;
                note.position = 0;
                notes.push_back(note);
            }
            cout << staffNumber << endl;
		}

    for (int i = 0; i < notes.size(); i++) {
		cout << "Note " << i << ": " << notes[i].name << endl;
	}

    return notes;
}

void binaryImage(Mat image, Mat& binaryImage) {
    binaryImage = Mat(image.rows, image.cols, CV_8UC1, Scalar(0)); // Initialize binaryImage with zeros

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if (image.at<uchar>(i, j) < 128)
                binaryImage.at<uchar>(i, j) = 0;
            else
                binaryImage.at<uchar>(i, j) = 255;
        }
    }
}

void drawNoteHeads(Mat& image, const vector<Point>& noteCenters, int radius = 3, const Scalar& color = Scalar(0, 0, 255)) {
    for (const auto& center : noteCenters) {
        circle(image, center, radius, color, FILLED); // Draw filled circles at each center of mass
    }
}

void adjustNoteTails(Mat& binaryImage, int threshold, int adjustment) {
    vector<int> verticalProjection = calculateVerticalProjection(binaryImage);
    for (int j = 0; j < binaryImage.cols; ++j) {
        if (verticalProjection[j] > threshold) {
            //cout << "Adjusting pixel values to the right at column " << j << endl;
            
            for (int i = 0; i < binaryImage.rows; ++i) {
                if (binaryImage.at<uchar>(i, j) == 0) {
                    for (int k = 1; k <= adjustment; ++k) {
                        if (j + k < binaryImage.cols) {
							binaryImage.at<uchar>(i, j + k) = 0;
						}
					}
				}
			}
        }
    }
}

void writeNotesAndDurationsToFile(const vector<Note>& notes) {
    ofstream outputFile("notes.txt");
    if (!outputFile.is_open()) {
        cout << "Error opening file for writing notes." << endl;
        return;
    }

    for (int i = 0; i < notes.size(); ++i) {
        char noteChar;
        switch (notes[i].name) {
        case C: noteChar = 'C'; break;
        case D: noteChar = 'D'; break;
        case E: noteChar = 'E'; break;
        case F: noteChar = 'F'; break;
        case G: noteChar = 'G'; break;
        case A: noteChar = 'A'; break;
        case B: noteChar = 'B'; break;
        }

        outputFile << noteChar << notes[i].octave;

        outputFile << " ";
        switch (notes[i].duration) {
        case quarter: outputFile << "4"; break;
        case eighth: outputFile << "8"; break;
        }

        outputFile << endl;
    }

    outputFile.close();
    cout << "Notes and durations written to file successfully." << endl;
}

vector<pair<int, int>> calculateAreaForDuration(Mat labels, Mat labelsNotes) {
    vector<pair<int, int>> areas;
    for (int label = 1; label <= newLabel; ++label) {
        int area = 0;

        // Check if any part of labelsNotes overlaps with the current label in labels
        bool overlap = false;

        for (int i = 0; i < labels.rows; ++i) {
            for (int j = 0; j < labels.cols; ++j) {
                if (labels.at<int>(i, j) == label && labelsNotes.at<int>(i, j) != 0) {
                    overlap = true;
                    area++;
                }
            }
        }

        areas.push_back({ label, area });

        if (overlap) {
            cout << "Area for label " << label << " overlapped with notes: " << area << endl;
            if ((area > 200 || area < 15) || (area > 90 && area < 100)) {
                cout << "Adjusting pixel values for label " << label << endl;
				for (int i = 0; i < labels.rows; ++i) {
					for (int j = 0; j < labels.cols; ++j) {
						if (labels.at<int>(i, j) == label) {
							labels.at<int>(i, j) = 0;
						}
					}
				}
			}
        }
        else {
            cout << "No overlap found for label " << label << endl;
            
            for (int i = 0; i < labels.rows; ++i) {
                for (int j = 0; j < labels.cols; ++j) {
                    if (labels.at<int>(i, j) == label) {
                        labels.at<int>(i, j) = 0;
                    }
                }
            }
        }
    }

    return areas;
}

vector<Note> identifyDurations(const Mat& labels, const vector<Staff>& staffs, Mat image, const Mat& labelsNotes, vector<Note> notes) {
    vector<Note> notesClone = notes;
    vector<Duration> durations;
    vector<pair<Point, int>> sortedLabels;

    vector<pair<int, int>> areas = calculateAreaForDuration(labels, labelsNotes);
    cout << "am ajuns aici" << endl;
    sortLabelsByCenterOfMass(labels, staffs, sortedLabels, image);

    //extract sorted labels
    vector<int> extractedLabels;
    cout << sortedLabels.size() << endl;
    for (const auto& pair : sortedLabels) {
        extractedLabels.push_back(pair.second);
    }

    // Display the sorted labels
    for (int i = 0; i < extractedLabels.size(); i++) {
        Point center = sortedLabels[i].first;
        //cout << "Center of mass for label " << extractedLabels[i] << ": " << center << endl;
        circle(image, center, 3, Scalar(0, 0, 255), FILLED);
        imshow("Detected Notes", image);
        waitKey(0);
    }

    auto isWhite = [](int pixel) { return pixel == 255; };

    int tolerance = 10;

    // for (int i = 0; i < extractedLabels.size(); i++) {
    //     int label = extractedLabels[i];

    //     int y = 0;

    //     for (int iN = 0; iN < labelsNotes.rows; iN++) {
    //         for (int jN = 0; jN < labelsNotes.cols; jN++) {
    //             if (labelsNotes.at<int>(iN, jN) == label) {
    //                 y = jN;
             //	}
             //}
    //     }

    //     circle(image, Point(label, y), 3, Scalar(0, 255, 0), FILLED);
    //     imshow("Detected Labels", image);
    //     waitKey(0);

    //     int upCount = 0;
    //     int j = y - 1;
    //     while (j >= 0 && !isWhite(labels.at<int>(j, label))) {
    //         upCount++;
    //         j--;
    //     }

    //     int downCount = 0;
    //     j = y + 1;
    //     while (j < labels.rows && !isWhite(labels.at<int>(j, label))) {
    //         downCount++;
    //         j++;
    //     }

    //     int rightCount = 0;
    //     j = label + 1; 
    //     bool firstWhitePixelFound = false;
    //     while (j < labels.cols && (!firstWhitePixelFound || !isWhite(labels.at<int>(y, j)))) {
    //         if (!firstWhitePixelFound && isWhite(labels.at<int>(y, j))) {
    //             firstWhitePixelFound = true;
    //             break;
    //         }
    //         if (firstWhitePixelFound) {
    //             rightCount++;
    //         }
    //         j++; // Increment the column index
    //     }

    //     cout << "Up count: " << upCount << ", Down count: " << downCount << ", Right count: " << rightCount << "Label value" << labels.at<int>(j, label) << endl;
    //     circle(image, Point(label, y), 3, Scalar(0, 255, 0), FILLED);
    //     imshow("Detected Labels", image);
    //     waitKey(0);

    //     if (upCount > tolerance) {
    //         if (rightCount > tolerance) {
    //             int thresholdColumns = 10;
    //             bool hasNonWhiteInThreshold = false;
    //             for (int k = 1; k <= thresholdColumns; ++k) {
    //                 if (!isWhite(labels.at<int>(y + k, label))) {
    //                     hasNonWhiteInThreshold = true;
    //                     break;
    //                 }
    //             }

    //             if (hasNonWhiteInThreshold) {
    //                 durations.push_back(eighth);
    //             }
    //             else {
    //                 durations.push_back(quarter);
    //             }
    //         }
    //         else {
    //             durations.push_back(quarter);
    //         }
    //     }
    //     else {
    //         durations.push_back(quarter);
    //     }

    // }

    // for (int i = 0; i < durations.size(); i++) {
    //     cout << "Duration " << i << ": " << durations[i] << endl;
    // }

    cout << "Areas size: " << areas.size() << endl;


    for (int i = 0; i < areas.size(); i++) {
        cout << "Area " << i << ": " << areas[i].second << endl;
        if (areas[i].second != 0) {
            if (areas[i].second > 25 && areas[i].second < 45) {
                for (int label = 0; label < notesClone.size(); label++) {
                    if (areas[i].first == label) {
						notesClone[label].duration = quarter;
						durations.push_back(quarter);
					}
                }
            }
            else {
                for (int label = 0; label < notesClone.size(); label++) {
                    if (areas[i].first == label) {
                        notesClone[label].duration = eighth;
                        durations.push_back(eighth);
                    }
                }
            }
        }
	}

    for (int i = 0; i < durations.size(); i++) {
		cout << "Duration " << i << ": " << durations[i] << endl;
    }

	return notesClone;
}

int main() {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    Mat image1 = imread("varianta_noua.png", 0);

    Mat image;

    binaryImage(image1, image);

    if (image.empty()) {
        cout << "Could not open or find the image" << endl;
        return -1;
    }

    // Detect portatives
    vector<Staff> staffs = detectStaffs(image, 1);

    imshow("Detected Portatives", image);

    Mat_<uchar> strel3(3, 3);
    strel3(0, 0) = 0;
    strel3(0, 1) = 0;
    strel3(0, 2) = 0;
    strel3(1, 0) = 1;
    strel3(1, 1) = 1;
    strel3(1, 2) = 1;
    strel3(2, 0) = 0;
    strel3(2, 1) = 0;
    strel3(2, 2) = 0;
    Mat clonee = erosionLab7(image, strel3);
    imshow("Eroded Portative Clone", clonee);
    waitKey(0);

    adjustNoteTails(image, 35, 2);

    Mat_<uchar> strel1(2, 2);
    strel1.setTo(0);
    Mat erodedPortative = openingLab7(image, strel1);

    imshow("Eroded Portative", erodedPortative);
    waitKey(0);

    Mat dilatedSrc;
    Mat_<uchar> strel2(5, 5);
    strel2.setTo(0);
    dilatedSrc = closingLab7(erodedPortative, strel2);

    imshow("Dilated Portative", dilatedSrc);
    waitKey(0);

    Mat labels = connectedLabeling(clonee);

    Mat colorImg(clonee.rows, clonee.cols, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < clonee.rows; i++) {
        for (int j = 0; j < clonee.cols; j++) {
            if (labels.at<int>(i, j) != 0) {
				colorImg.at<Vec3b>(i, j) = Vec3b(rand() % 256, rand() % 256, rand() % 256);
			}
		}
	}

    vector<Note> notes = identifyNotes(labels, staffs, colorImg);

    Mat labelsDur = connectedLabeling(dilatedSrc);

    cout << "Connected labeling for durations" << endl;

    Mat colorImgForDurations(dilatedSrc.rows, dilatedSrc.cols, CV_8UC3, Scalar(255, 255, 255));
    for (int i = 0; i < dilatedSrc.rows; i++) {
        for (int j = 0; j < dilatedSrc.cols; j++) {
            if (labelsDur.at<int>(i, j) != 0) {
                colorImgForDurations.at<Vec3b>(i, j) = Vec3b(rand() % 256, rand() % 256, rand() % 256);
            }
        }
    }

    vector<Note> durations = identifyDurations(labelsDur, staffs, colorImgForDurations, labels, notes);

    writeNotesAndDurationsToFile(notes);

    return 0;
}
