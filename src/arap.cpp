#include "arap.h"
#include "graphics/meshloader.h"

#include <iostream>
#include <set>
#include <map>
#include <vector>
#include <Eigen/Sparse>

using namespace std;
using namespace Eigen;

ARAP::ARAP() {}

void ARAP::init(Eigen::Vector3f &coeffMin, Eigen::Vector3f &coeffMax)
{
    vector<Vector3f> vertices;
    vector<Vector3i> triangles;

    // If this doesn't work for you, remember to change your working directory
    if (MeshLoader::loadTriMesh("meshes/peter.obj", vertices, triangles)) {
        m_shape.init(vertices, triangles);

        vertexToNeighbor.clear();
        edgeToCotWeight.clear();

        #pragma omp parallel for
        for (const Eigen::Vector3i &triangle : triangles) {
            #pragma omp parallel for
            for (int i = 0; i < 3; ++i) {
                int vertexIndex = triangle[i];
                // Add the other two vertices of the triangle to the neighbor list of vertexIndex
                for (int j = 0; j < 3; ++j) {
                    if (i != j) {
                        int neighborIndex = triangle[j];
                        // Add the position of the neighbor to the list, if it's not already there
                        Eigen::Vector3f neighborPos = vertices[neighborIndex];
                        if (std::find(vertexToNeighbor[vertexIndex].begin(), vertexToNeighbor[vertexIndex].end(), neighborIndex) == vertexToNeighbor[vertexIndex].end()) {
                            vertexToNeighbor[vertexIndex].push_back(neighborIndex);
                        }
                    }
                }
            }
        }

        initial_vertices = vertices;

        ori_anchors = m_shape.getAnchors();

        // Compute cotangent weights and set up the Laplacian matrix
        computeCotangentWeight();
        setLMatrix();
    }

    // Students, please don't touch this code: get min and max for viewport stuff
    MatrixX3f all_vertices = MatrixX3f(vertices.size(), 3);
    int i = 0;
    for (unsigned long i = 0; i < vertices.size(); ++i) {
        all_vertices.row(i) = vertices[i];
    }
    coeffMin = all_vertices.colwise().minCoeff();
    coeffMax = all_vertices.colwise().maxCoeff();
}

// Move an anchored vertex, defined by its index, to targetPosition
void ARAP::move(int vertex, Vector3f targetPosition)
{
    std::vector<Eigen::Vector3f> new_vertices = m_shape.getVertices();
    const std::unordered_set<int>& anchors = m_shape.getAnchors();

    // TODO: implement ARAP here
    new_vertices[vertex] = targetPosition;

    // Here are some helpful controls for the application
    //
    // - You start in first-person camera mode
    //   - WASD to move, left-click and drag to rotate
    //   - R and F to move vertically up and down
    //
    // - C to change to orbit camera mode
    //
    // - Right-click (and, optionally, drag) to anchor/unanchor points
    //   - Left-click an anchored point to move it around
    //
    // - Minus and equal keys (click repeatedly) to change the size of the vertices

    m_shape.setVertices(new_vertices);

    int iter = 0;
    float bestenergy = std::numeric_limits<float>::infinity();
    std::vector<Eigen::Vector3f> best_est;
    while (iter < 1){
        // Compute the best-fit rotation matrices R
        computeBestFitRotations(new_vertices);

        // Optimize the positions p' of the vertices
        float energy = optimizePositions(new_vertices);

        if (energy < bestenergy){
            bestenergy = energy;
            best_est = curr_est;
        }

        iter += 1;
    }

    m_shape.setVertices(best_est);

}

void ARAP::update(){
    if (ori_anchors != m_shape.getAnchors()){
        ori_anchors = m_shape.getAnchors();
        setLMatrixConstraint();
    }
    else if (initial_vertices != m_shape.getVertices()){
        edgeToCotWeight.clear();
        initial_vertices = m_shape.getVertices();
        computeCotangentWeight();
        setLMatrix();
    }
}

void ARAP::computeCotangentWeight(){
    std::vector<Eigen::Vector3i> curr_triangles = m_shape.getFaces();
    std::vector<Eigen::Vector3f> curr_vertices = m_shape.getVertices();
    #pragma omp parallel for
    for (const Eigen::Vector3i &triangle : curr_triangles) {
        for (int i = 0; i < 3; ++i) {
            int vertexIndexA = triangle[i];
            int vertexIndexB = triangle[(i + 1) % 3];
            int vertexIndexC = triangle[(i + 2) % 3];

            Eigen::Vector3f vertexA = curr_vertices[vertexIndexA];
            Eigen::Vector3f vertexB = curr_vertices[vertexIndexB];
            Eigen::Vector3f vertexC = curr_vertices[vertexIndexC];

            float lengthA = (vertexC - vertexB).norm();
            float lengthB = (vertexC - vertexA).norm();
            float lengthC = (vertexB - vertexA).norm();

            // Cotangent for angle at vertex C
            float cosC = (lengthA * lengthA + lengthB * lengthB - lengthC * lengthC) / (2 * lengthA * lengthB);
            float sinC = std::sqrt(1 - cosC * cosC);
            float cotC = cosC / sinC;

            // Update the cotangent weight for the edge (vertexIndexA, vertexIndexB)
            std::pair<int, int> edgeAB = std::minmax(vertexIndexA, vertexIndexB);
            edgeToCotWeight[edgeAB] += 0.5f * abs(cotC);
        }
    }
}

void ARAP::setLMatrix(){
    int numVertices = m_shape.getVertices().size();
    Eigen::SparseMatrix<float> L(numVertices, numVertices);

    #pragma omp parallel for
    for (const auto &edgeWeightPair : edgeToCotWeight) {
        int i = edgeWeightPair.first.first;
        int j = edgeWeightPair.first.second;
        float weight = edgeWeightPair.second;

        // Update the off-diagonal entries
        L.coeffRef(i, j) = -weight;
        L.coeffRef(j, i) = -weight;

        // Update the diagonal entries
        L.coeffRef(i, i) += weight;
        L.coeffRef(j, j) += weight;
    }

    // Apply user constraints
    const std::unordered_set<int>& anchors = m_shape.getAnchors();
    #pragma omp parallel for
    for (int anchor : anchors) {
        L.coeffRef(anchor, anchor) = 1;

        // Set all off-diagonal entries in the row and column to zero
        #pragma omp parallel for
        for (int k = 0; k < numVertices; ++k) {
            if (k != anchor) {
                L.coeffRef(anchor, k) = 0;
                L.coeffRef(k, anchor) = 0;
            }
        }
    }

    m_LMatrix = L;

    // std::cout << L;

    // Precompute the decomposition of the L matrix
    m_LDecomposition.compute(m_LMatrix);

    // Check if the decomposition succeeded
    // if (m_LDecomposition.info() != Eigen::Success) {
    //     throw std::runtime_error("Failed to decompose the L matrix");
    // }
}

void ARAP::setLMatrixConstraint(){
    int numVertices = m_shape.getVertices().size();

    // Apply user constraints
    const std::unordered_set<int>& anchors = m_shape.getAnchors();
    #pragma omp parallel for
    for (int anchor : anchors) {
        m_LMatrix.coeffRef(anchor, anchor) = 1;

        // Set all off-diagonal entries in the row and column to zero
    #pragma omp parallel for
        for (int k = 0; k < numVertices; ++k) {
            if (k != anchor) {
                m_LMatrix.coeffRef(anchor, k) = 0;
                m_LMatrix.coeffRef(k, anchor) = 0;
            }
        }
    }

    // Precompute the decomposition of the L matrix
    m_LDecomposition.compute(m_LMatrix);
}

void ARAP::computeBestFitRotations(std::vector<Eigen::Vector3f> deformedVertices) {
    int numVertices = initial_vertices.size();
    m_rotations.resize(numVertices);

    #pragma omp parallel for
    for (int i = 0; i < numVertices; ++i) {
        Eigen::Matrix3f Si = Eigen::Matrix3f::Zero();
        #pragma omp parallel for
        for (int neighborIndex : vertexToNeighbor[i]) {
            Eigen::Vector3f pi = initial_vertices[i];
            Eigen::Vector3f pj = initial_vertices[neighborIndex];
            Eigen::Vector3f piPrime = deformedVertices[i];
            Eigen::Vector3f pjPrime = deformedVertices[neighborIndex];

            // Calculate the original and deformed edge vectors
            Eigen::Vector3f eij = pi - pj;
            Eigen::Vector3f eijPrime = piPrime - pjPrime;

            // Retrieve the cotangent weight for the edge
            float wij = edgeToCotWeight[std::minmax(i, neighborIndex)];

            // Add the weighted outer product to Si
            Si += wij * eij * eijPrime.transpose();
        }

        // SVD to get rotation
        Eigen::JacobiSVD<Eigen::Matrix3f> svd(Si, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3f Ui = svd.matrixU();
        Eigen::Matrix3f Vi = svd.matrixV();

        // Correct for potential reflection
        Eigen::Matrix3f Ri = Vi * Ui.transpose();
        if (Ri.determinant() < 0) {
            Eigen::Matrix3f UiCorrected = Ui;
            UiCorrected.col(2) *= -1;
            Ri = Vi * UiCorrected.transpose();
        }

        // Store the rotation matrix
        m_rotations[i] = Ri;

    }

}

float ARAP::optimizePositions(std::vector<Eigen::Vector3f> deformedVertices) {
    int numVertices = initial_vertices.size();
    Eigen::MatrixXf b = Eigen::MatrixXf::Zero(numVertices, 3);

    const std::unordered_set<int>& anchors = m_shape.getAnchors();

    // Update the right-hand side b based on the rotations and original positions
    #pragma omp parallel for
    for (int i = 0; i < numVertices; ++i) {
        Eigen::Vector3f sumRiPij(0, 0, 0);
        for (int j : vertexToNeighbor[i]) {
            float weight = edgeToCotWeight[std::minmax(i, j)];
            Eigen::Vector3f pij = initial_vertices[i] - initial_vertices[j];
            sumRiPij += 0.5 * weight * (m_rotations[i] + m_rotations[j]) * pij;
        }
        b.row(i) = sumRiPij;
    }

    applyConstraints(deformedVertices, b);

    // Solve the linear system Lp' = b
    // Eigen::MatrixXf pPrime = m_LDecomposition.solve(b);
    // Define pPrime as a matrix of floats
    Eigen::MatrixXf pPrime(numVertices, 3);

// Perform the solve operation in parallel for each coordinate
#pragma omp parallel sections
    {
#pragma omp section
        {
            // Solve for the x-coordinates
            pPrime.col(0) = m_LDecomposition.solve(b.col(0));
        }
#pragma omp section
        {
            // Solve for the y-coordinates
            pPrime.col(1) = m_LDecomposition.solve(b.col(1));
        }
#pragma omp section
        {
            // Solve for the z-coordinates
            pPrime.col(2) = m_LDecomposition.solve(b.col(2));
        }
    }

    // Update the deformed vertex positions in the shape
    std::vector<Eigen::Vector3f> new_vertices;
    float energy = 0;
    #pragma omp parallel for
    for (int i = 0; i < numVertices; ++i) {
        new_vertices.push_back(pPrime.row(i));
        float sum = 0;
        for (int j : vertexToNeighbor[i]){
            float weight = edgeToCotWeight[std::minmax(i, j)];
            sum += weight * ((pPrime.row(i).transpose() - pPrime.row(j).transpose()) - m_rotations[i] * (initial_vertices[i] - initial_vertices[j])).squaredNorm();
        }
        energy += sum;
    }

    curr_est = new_vertices;
    return energy;
}


void ARAP::applyConstraints(std::vector<Eigen::Vector3f> deformedVertices, Eigen::MatrixXf &b) {
    const std::unordered_set<int>& anchors = m_shape.getAnchors();
    #pragma omp parallel for
    for (int anchor : anchors) {
        for (int neighbor: vertexToNeighbor[anchor]){
            float weight = edgeToCotWeight[std::minmax(anchor, neighbor)];
            b.row(neighbor) += weight * m_shape.getVertices()[anchor];
        }
    }

    #pragma omp parallel for
    for (int anchor : anchors){
        b.row(anchor) = m_shape.getVertices()[anchor];
    }

}
