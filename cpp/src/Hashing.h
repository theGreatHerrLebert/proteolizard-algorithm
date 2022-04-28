#ifndef SRC_HASHING_H
#define SRC_HASHING_H

#include <Eigen/Dense>
#include <ProteoLizardData/Frame.h>
#include <set>

// helper to check for self collision of windows
class CollisionBox{
public:
    CollisionBox() = default;
    CollisionBox(int bin, int hashKey, int scan);
    void append(int hashKey, int scan);
    std::pair<int, std::set<int>> flatCollisions();

    // mz bin of box
    int mzBin {};
    // data structure to keep track of hash-key, collision status and scans
    std::map<int, std::pair<bool, std::vector<int>>> keyMap {};
};

class TimsHashGenerator {
public:
    TimsHashGenerator(int kk, int ll, int s, int r, int lenDalton);

    const Eigen::MatrixXd &getMatrixCopy() { return M; }

    std::pair<std::vector<int>, std::vector<std::vector<int>>> hashSpectrum(MzSpectrumPL &spectrum,
                                                                            int minPeaksPerWindow,
                                                                            int minIntensity,
                                                                            double  windowLength,
                                                                            bool overlapping,
                                                                            bool binRestricted);

    std::vector<int> hashSpectrum(MzSpectrumPL &spectrum);

    std::pair<std::vector<int>, std::pair<std::vector<int>, std::vector<std::vector<int>>>> hashFrame(
            TimsFramePL &frame,
            int minPeaksPerWindow,
            int minIntensity,
            double windowLength,
            bool overlapping,
            bool binRestricted);

    std::pair<std::vector<int>, std::vector<int>> getCollisionInBands(Eigen::MatrixXi H, std::vector<int> scans, std::vector<int> bins);

    int seed;
    int resolution;
    int k;
    int l;
    const Eigen::MatrixXd M;
};

#endif //SRC_HASHING_H
