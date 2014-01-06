#include <iostream>
#include <string>
#include <cstdio>
#include <vector>
#include <list>
#include <algorithm>
using namespace std;

#include "utils.h"
#include "cudaProj.h"
using namespace tcs_cuda;

#define COMMAND_SET_INPUT 2
#define COMMAND_SET_OUTPUT 3
#define COMMAND_EXIT 4

string inputFileName = "input.png";
string outputFileName = "output.png";

typedef void (*Filter)(const GPUImage&, GPUImage&);
typedef GPUImage (*Transform)(const GPUImage&);

GPUImage defaultTransform(const GPUImage& input) {
    return GPUImage::createEmpty(input.getWidth(), input.getHeight());
}

class MenuNode {
    private:
        MenuNode            *_parent;
        string              _label;
        vector<MenuNode* >  _children;

        Filter              _filter;
        Transform           _transform;

    public:
        static MenuNode     *rootNode;
        
        MenuNode(MenuNode* parent, const string& label, Filter filter, Transform transform) 
            : _label(label), _filter(filter), _parent(parent), _transform(transform) {}

        ~MenuNode() {
            for (vector<MenuNode*>::iterator it = _children.begin(); it != _children.end(); it++) {
                delete *it;
                *it = 0;
            }
            _parent = 0;
        }

        MenuNode* goToLabel(const string& label) {
            for (vector<MenuNode*>::iterator it = _children.begin(); it != _children.end(); it++) {
                if ((*it)->_label == label)
                    return *it;
            }
            MenuNode *node = new MenuNode(this, label, 0, 0);
            _children.push_back(node);
            return node;
        }

        MenuNode* makeLeaf(const string& label, Filter func, Transform transform) {
            MenuNode *node = new MenuNode(this, label, func, transform);
            _children.push_back(node);
            return node;
        }

        void showMenu() {
            system("clear");
            printf(" Select an option: \n");

            int pos = 2;
            printf(" 1 -> ..\n");
            for (vector<MenuNode*>::const_iterator it = _children.begin(); it != _children.end(); it++) {
                MenuNode *node = *it;
                if (node->_children.size() > 0) //directory
                    printf(" %d -> %s/\n", pos++, node->_label.c_str());
                else 
                    printf(" %d -> %s\n", pos++, node->_label.c_str());
            }

            printf("----------------------\n");
            printf(" %d -> set input file\n", pos++);
            printf(" %d -> set output file\n", pos++);
            printf(" %d -> exit program\n", pos++);

            printf("----------------------\n");
            printf("input file: %s\n", inputFileName.c_str());
            printf("output file: %s\n", outputFileName.c_str());
            
            printf("\n>");
        }

        MenuNode* enterCommand(int command) {

            //wrong command
            if (command < 1 || command > _children.size() + COMMAND_EXIT)
                return this;

            //go menu up
            if (command == 1) 
                return (_parent == 0) ? this : _parent;

            //set input
            if (command == _children.size() + COMMAND_SET_INPUT) {
                system("clear");
                char tmp[1024];
                printf("Enter input file name: ");
                scanf("%s", tmp);
                inputFileName = string(tmp);
                return rootNode;
            }

            //set output
            if (command == _children.size() + COMMAND_SET_OUTPUT) {
                system("clear");
                char tmp[1024];
                printf("Enter output file name: ");
                scanf("%s", tmp);
                outputFileName = string(tmp);
                return rootNode;
            }

            //exit program
            if (command == _children.size() + COMMAND_EXIT) return 0; //exit the program
            
            //enter node
            MenuNode *next = _children[command-2];
            if (next->_filter != 0) {
                GPUImage inputImage = GPUImage::load(inputFileName);
                GPUImage outputImage = next->_transform(inputImage);
                next->_filter(inputImage, outputImage);
                GPUImage::save(outputFileName, outputImage);
                return MenuNode::rootNode;
            }

            return next;
        }
};

MenuNode* MenuNode::rootNode = 0;

void registerFilter(const string& path, Filter filter, Transform transform = &defaultTransform) {
    if(path[0] != '.') return;
    if ("." == path){ //rootNode
        MenuNode::rootNode = new MenuNode(0, ".", 0, 0);
        return;
    }
    
    int len = path.length(), ind = 1;
    string lastLabel = "";
    MenuNode *currentNode = MenuNode::rootNode;

    while (ind < len) {
        lastLabel = "";
        while (ind < len && path[ind] != '.') {
            lastLabel += path[ind++];
        }
        if (ind < len) //node
            currentNode = currentNode->goToLabel(lastLabel);
        else { // leaf
            currentNode->makeLeaf(lastLabel, filter, transform);
            return;
        }
        ind++;
    }
}

void registerFilters() {
    //root
    registerFilter(".", 0, 0);

    //invert
    registerFilter(".PerPixel.Invert", projInvert);

    //matrix
    registerFilter(".Matrix.Embos.EmbosEast", projMatrix3x3_EmbosEast);
    registerFilter(".Matrix.Embos.EmbosSouthEast", projMatrix3x3_EmbosSouthEast);
    registerFilter(".Matrix.Embos.EmbosSouth", projMatrix3x3_EmbosSouth);
    registerFilter(".Matrix.Embos.EmbosSouthWest", projMatrix3x3_EmbosWest);
    registerFilter(".Matrix.Embos.EmbosWest", projMatrix3x3_EmbosWest);
    registerFilter(".Matrix.Embos.EmbosNorthWest", projMatrix3x3_EmbosNorthWest);
    registerFilter(".Matrix.Embos.EmbosNorth", projMatrix3x3_EmbosNorth);
    registerFilter(".Matrix.Embos.EmbosNorthEast", projMatrix3x3_EmbosNorthEast);
    registerFilter(".Matrix.HighPass.MeanRemoval", projMatrix3x3_MeanRemoval);
    registerFilter(".Matrix.HighPass.HP1", projMatrix3x3_HighPass1);
    registerFilter(".Matrix.HighPass.HP2", projMatrix3x3_HighPass2);
    registerFilter(".Matrix.HighPass.HP3", projMatrix3x3_HighPass3);
    registerFilter(".Matrix.LowPass.Average", projMatrix3x3_Average);
    registerFilter(".Matrix.LowPass.LP1", projMatrix3x3_LowPass1);
    registerFilter(".Matrix.LowPass.LP2", projMatrix3x3_LowPass2);
    registerFilter(".Matrix.LowPass.LP3", projMatrix3x3_LowPass3);
    registerFilter(".Matrix.LowPass.Gauss", projMatrix3x3_LowPassGauss);
    registerFilter(".Matrix.EdgeDetection.Gradient.East", projMatrix3x3_EdgeDetection_GradientEast);
    registerFilter(".Matrix.EdgeDetection.Gradient.SouthEast", projMatrix3x3_EdgeDetection_GradientSouthEast);
    registerFilter(".Matrix.EdgeDetection.Gradient.South", projMatrix3x3_EdgeDetection_GradientSouth);
    registerFilter(".Matrix.EdgeDetection.Gradient.SouthWest", projMatrix3x3_EdgeDetection_GradientSouthWest);
    registerFilter(".Matrix.EdgeDetection.Gradient.West", projMatrix3x3_EdgeDetection_GradientWest);
    registerFilter(".Matrix.EdgeDetection.Gradient.NorthWest", projMatrix3x3_EdgeDetection_GradientNorthWest);
    registerFilter(".Matrix.EdgeDetection.Gradient.North", projMatrix3x3_EdgeDetection_GradientNorth);
    registerFilter(".Matrix.EdgeDetection.Gradient.NorthEast", projMatrix3x3_EdgeDetection_GradientNorthEast);
    registerFilter(".Matrix.EdgeDetection.Laplace.L1", projMatrix3x3_EdgeDetection_Laplace1);
    registerFilter(".Matrix.EdgeDetection.Laplace.L2", projMatrix3x3_EdgeDetection_Laplace2);
    registerFilter(".Matrix.EdgeDetection.Laplace.L3", projMatrix3x3_EdgeDetection_Laplace3);
    registerFilter(".Matrix.EdgeDetection.Laplace.Diagonal", projMatrix3x3_EdgeDetection_LaplaceDiagonal);
    registerFilter(".Matrix.EdgeDetection.Laplace.Vertical", projMatrix3x3_EdgeDetection_LaplaceVertical);
    registerFilter(".Matrix.EdgeDetection.Laplace.Horizontal", projMatrix3x3_EdgeDetection_LaplaceHorizontal);
    registerFilter(".Matrix.EdgeDetection.Prewitt.Vertical", projMatrix3x3_EdgeDetection_PrewittVertical);
    registerFilter(".Matrix.EdgeDetection.Prewitt.Horizontal", projMatrix3x3_EdgeDetection_PrewittHorizontal);
    registerFilter(".Matrix.EdgeDetection.Sobel.Vertical", projMatrix3x3_EdgeDetection_SobelVertical);
    registerFilter(".Matrix.EdgeDetection.Sobel.Horizontal", projMatrix3x3_EdgeDetection_SobelHorizontal);
    registerFilter(".Matrix.EdgeDetection.Standard.Vertical", projMatrix3x3_EdgeDetection_Vertical);
    registerFilter(".Matrix.EdgeDetection.Standard.Horizontal", projMatrix3x3_EdgeDetection_Horizontal);
    registerFilter(".Matrix.EdgeDetection.Standard.Diagonal1", projMatrix3x3_EdgeDetection_Diagonal1);
    registerFilter(".Matrix.EdgeDetection.Standard.Diagonal2", projMatrix3x3_EdgeDetection_Diagonal2);

    //histogram
    registerFilter(".Histogram.All", projHistogram_All, histogramTransform);
    registerFilter(".Histogram.Red", projHistogram_Red, histogramTransform);
    registerFilter(".Histogram.Green", projHistogram_Green, histogramTransform);
    registerFilter(".Histogram.Blue", projHistogram_Blue, histogramTransform);

    //transform
    registerFilter(".Transform.Flip.Horizontal", flipHor); 
    registerFilter(".Transform.Flip.Vertical", flipVer); 
    registerFilter(".Transform.Rotate.Left", rotateLeft, rotateTransform); 
    registerFilter(".Transform.Rotate.Right", rotateRight, rotateTransform); 
    // registerFilter(".Transform.Rescale", rescale); 
    
}

int main() {

    registerFilters();
    if (MenuNode::rootNode == 0) {
        printf("Filter registration failed!\n");
        exit(1);
    }

    MenuNode *node = MenuNode::rootNode;
    while (node != 0) {
        node->showMenu();
        int command; scanf("%d", &command);
        node = node->enterCommand(command);
    }

    system("clear");
    delete MenuNode::rootNode;  
    return 0;
}
