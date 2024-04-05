#pragma once

#include "graphics/shape.h"
#include "Eigen/StdList"
#include "Eigen/StdVector"
#include "map"
#include <Eigen/Sparse>

class Shader;

class ARAP
{
private:
    Shape m_shape;

public:
    ARAP();

    void init(Eigen::Vector3f &min, Eigen::Vector3f &max);
    void move(int vertex, Eigen::Vector3f pos);
    void computeCotangentWeight();
    void setLMatrix();
    void computeBestFitRotations(std::vector<Eigen::Vector3f> deformedVertices);
    void applyLMatrixConstraints(Eigen::SparseMatrix<float> &L);
    float optimizePositions(std::vector<Eigen::Vector3f> deformedVertices);
    void applyConstraints(std::vector<Eigen::Vector3f> deformedVertices, Eigen::MatrixXf &b);
    void update();

    std::vector<Eigen::Vector3f> initial_vertices;
    std::map<int, std::vector<int>> vertexToNeighbor;
    std::map<std::pair<int, int>, float> edgeToCotWeight;
    Eigen::SparseMatrix<float> m_LMatrix;
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> m_LDecomposition;
    std::vector<Eigen::Matrix3f> m_rotations;
    std::unordered_set<int> ori_anchors;
    std::vector<Eigen::Vector3f> curr_est;
    bool startmoving = false;

    // ================== Students, If You Choose To Modify The Code Below, It's On You

    int getClosestVertex(Eigen::Vector3f start, Eigen::Vector3f ray, float threshold)
    {
        return m_shape.getClosestVertex(start, ray, threshold);
    }

    void draw(Shader *shader, GLenum mode)
    {
        m_shape.draw(shader, mode);
    }

    SelectMode select(Shader *shader, int vertex)
    {
        return m_shape.select(shader, vertex);
    }

    bool selectWithSpecifiedMode(Shader *shader, int vertex, SelectMode mode)
    {
        return m_shape.selectWithSpecifiedMode(shader, vertex, mode);
    }

    bool getAnchorPos(int lastSelected, Eigen::Vector3f& pos, Eigen::Vector3f ray, Eigen::Vector3f start)
    {
        return m_shape.getAnchorPos(lastSelected, pos, ray, start);
    }
};
