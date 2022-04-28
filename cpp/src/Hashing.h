//
// Created by administrator on 06.04.22.
//

#ifndef SRC_HASHING_H
#define SRC_HASHING_H

#define assertm(exp, msg) assert(((void)msg, exp))

#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ProteoLizardData/Frame.h>
#include <cmath>
#include <algorithm>
#include <execution>
#include <set>

// helper to check for self collision of windows
struct CollisionBox{
    // mz bin of box
    int mzBin {};
    // data structure to keep track of hash-key, collision status and scans
    std::map<int, std::pair<bool, std::vector<int>>> keyMap {};

    CollisionBox()= default;
    CollisionBox(int bin, int hashKey, int scan);
    void append(int hashKey, int scan);
    std::pair<int, std::set<int>> flatCollisions();
};

CollisionBox::CollisionBox(int bin, int hashKey, int scan): mzBin(bin) {
    keyMap[hashKey] = {false, {scan}};
}


void CollisionBox::append(int hashKey, int scan) {
    if(this->keyMap.contains(hashKey)){
        auto &p = this->keyMap[hashKey];
        p.first = true;
        p.second.push_back(scan);
    } else {
        std::pair<bool, std::vector<int>> p = {false, {scan}};
        keyMap[hashKey] = p;
    }
}

std::pair<int, std::set<int>> CollisionBox::flatCollisions() {
    // TODO: check if length of retScan could be known a priori
    std::vector<int> retScans;
    for(auto &[key, pair]: this->keyMap){
        if (pair.first){
            retScans.insert(retScans.end(), pair.second.begin(), pair.second.end());
        }
    }
    std::set<int> s(retScans.begin(), retScans.end());
    return {this->mzBin, s};
}

std::map<int, std::set<int>> collisionsInBox(std::vector<int> band, std::vector<int> scans, std::vector<int> bins){

    // mz_bin -> CollisionBox
    std::map<int, CollisionBox> tmpMap;

    // go over all keys in band
    for(std::size_t i = 0; i < band.size(); i++){

        auto key = band[i];
        auto scan = scans[i];
        auto bin = bins[i];

        if(tmpMap.contains(bin)){
            tmpMap[bin].append(key, scan);
        }
        else {
            tmpMap[bin] = {bin, key, scan};
        }
    }

    std::map<int, std::set<int>> retMap;

    for(auto &[key, p]: tmpMap){

        auto flatCollision = p.flatCollisions();

        if(!flatCollision.second.empty())
            retMap[flatCollision.first] = flatCollision.second;
    }

    return retMap;
}


/**
 * check if a given key occurred in map more then once
 * @param keys
 * @param containsMap
 * @return
 */
bool containsKeyMultiple(const std::vector<int>& keys, const std::map<int, bool>& containsMap){
    bool moreThenOnce = false;
    for(const auto key: keys){
        if(containsMap.at(key)){
            moreThenOnce = true;
            break;
        }
    }
    return moreThenOnce;
}

/**
 * create eigen sparse vector from vectorized mz spectrum
 * @param mzVector : vectorized mz spectrum to convert
 * @param numRows : dimensionality of vector
 * @return : a sparse eigen vector suited for fast vectorized operations
 */
Eigen::SparseVector<double> toSparseVector(const MzVectorPL& mzVector, int numRows){

    Eigen::SparseMatrix<double> sparseVec = Eigen::SparseMatrix<double>(numRows, 1);
    std::vector<Eigen::Triplet<double>> tripletList;
    tripletList.reserve(mzVector.indices.size());

    for(std::size_t i = 0; i < mzVector.indices.size(); i++)
        tripletList.emplace_back(mzVector.indices[i], 0, mzVector.values[i]);

    sparseVec.setFromTriplets(tripletList.begin(), tripletList.end());
    return sparseVec;
}

/**
 * create a string representation of bits for easy hashing
 * @param boolVector vector to create bit string from
 * @return bit string
 */
std::string boolVectorToString(const std::vector<bool> &boolVector, int bin, bool restricted){

    std::string ret;

    for(const auto& b: boolVector)
        b ? ret.append("1") : ret.append("0");

    // hard restriction to its own mass bin only for collision
    if(restricted)
        ret.append(std::to_string(bin));
        // soft restriction to all windows with same offset
    else
        bin > 0 ? ret.append("1") : ret.append("0");

    return ret;
}

/**
 * calculate integer keys from vectors of boolean
 * @param hashes set of bool vectors representing the result from lsh probing
 * @return a set of ints representing the keyspace of a given mz spectrum or window
 */
std::vector<int> calculateKeys(const std::vector<std::vector<bool>> &hashes, int bin, bool restricted){
    std::vector<int> retVec;
    retVec.reserve(hashes.size());
    for(const auto& h: hashes){
        int hash = std::hash<std::string>{}(boolVectorToString(h, bin, restricted));
        retVec.push_back(hash);
    }
    return retVec;
}

/**
 *
 * @param sparseSpectrumVector
 * @param M random matrix of shape ((k * l) * mzSpace)
 * @param k number of ANDs
 * @param l number of ORs
 * @return k vectors of l bits
 */
std::vector<std::vector<bool>> calculateSignumVector(const Eigen::SparseVector<double>& sparseSpectrumVector,
                                                     const Eigen::MatrixXd& M,
                                                     int k,
                                                     int l){
    // check for compatible settings
    assertm(k * l == M.rows(), "dimensions of random vector matrix and banding differ.");

    std::vector<std::vector<bool>> retVec;
    retVec.reserve(k);

    // heavy lifting happens here, calculate dot products between random vectors and spectrum
    auto r = M * sparseSpectrumVector;

    // calculate signs from dot products
    std::vector<bool> bVec;
    bVec.reserve(l);

    for(std::size_t i = 0; i < r.size(); i++)
        bVec.push_back(r[i] > 0);

    // rest of the code is for grouping of results only
    std::vector<bool> leVec;
    leVec.reserve(l);

    for(std::size_t i = 0; i < bVec.size(); i++){
        if(i % l == 0 && i != 0){
            retVec.push_back(leVec);
            leVec.clear();
        }
        leVec.push_back(bVec[i]);
    }
    retVec.push_back(leVec);
    return retVec;
}

auto initMatrix = [](int k, int l, int seed, int res, int lenDalton) -> Eigen::MatrixXd {

    std::default_random_engine generator(seed);
    std::normal_distribution<double> distribution(0, 1);
    auto normal = [&] (int) {return distribution(generator);};

    int resFactor = int(pow(10, res));

    return Eigen::MatrixXd::NullaryExpr(k * l, lenDalton * resFactor + 1, normal);
};


struct TimsHashGenerator {

    int seed, resolution, k, l;
    const Eigen::MatrixXd M;

    TimsHashGenerator(int kk, int ll, int s, int r, int lenDalton): k(kk), l(ll), seed(s), resolution(r),
                                                                    M(initMatrix(kk, ll, s, r, lenDalton)){}

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
};



std::pair<std::vector<int>, std::vector<std::vector<int>>> TimsHashGenerator::hashSpectrum(
        MzSpectrumPL &spectrum,
        int minPeaksPerWindow,
        int minIntensityPerWindow,
        double windowLength,
        bool overlapping,
        bool binRestricted) {

    const auto windows = spectrum.windows(windowLength, overlapping, minPeaksPerWindow, minIntensityPerWindow);

    std::vector<std::pair<int, std::vector<int>>> retVec;
    retVec.resize(windows.size());

    std::map<int, Eigen::SparseVector<double>> tmpVec;
    int numRows = int(2000 * pow(10, this->resolution));

    auto hashWindow = [&binRestricted, &numRows, this](std::pair<const int, MzSpectrumPL> p) -> std::pair<int, std::vector<int>> {
        auto sparseVec = toSparseVector(p.second.vectorize(this->resolution), numRows);
        auto signumVec = calculateSignumVector(sparseVec, this->M, this->k, this->l);
        auto keys = calculateKeys(signumVec, p.first, binRestricted);
        return {p.first, keys};
    };

    std::transform(std::execution::par_unseq, windows.begin(), windows.end(), retVec.begin(), hashWindow);

    std::vector<int> retBins;
    std::vector<std::vector<int>> retHashes;

    retBins.reserve(retVec.size());
    retHashes.reserve(retVec.size());

    for(auto & p : retVec){
        retBins.push_back(p.first);
        retHashes.push_back(p.second);
    }

    return  {retBins, retHashes};
}

std::pair<std::vector<int>, std::pair<std::vector<int>, std::vector<std::vector<int>>>> TimsHashGenerator::hashFrame(
        TimsFramePL &frame,
        int minPeaksPerWindow,
        int minIntensity,
        double windowLength,
        bool overlapping,
        bool binRestricted){

    auto spectra = frame.spectra();

    std::vector<std::pair<int, std::pair<std::vector<int>, std::vector<std::vector<int>>>>> retVec;

    retVec.resize(spectra.size());

    auto hashSpectrum = [&binRestricted, &minPeaksPerWindow, &minIntensity, &windowLength,
            &overlapping, this] (std::pair<const int, MzSpectrumPL> p)
            -> std::pair<const int, std::pair<std::vector<int>, std::vector<std::vector<int>>>> {
        auto hashedSpectrum = this->hashSpectrum(
                p.second,
                minPeaksPerWindow,
                minIntensity,
                windowLength,
                overlapping,
                binRestricted);
        return {p.first, hashedSpectrum};
    };

    std::transform(std::execution::par_unseq, spectra.begin(), spectra.end(), retVec.begin(), hashSpectrum);

    std::vector<int> retScans;
    std::vector<int> retBins;
    std::vector<std::vector<int>> retHashes;

    for(auto& [scan, p]: retVec){
        retScans.insert(retScans.end(), p.first.size(), scan);
        retBins.insert(retBins.end(), p.first.begin(), p.first.end());
        retHashes.insert(retHashes.end(), p.second.begin(), p.second.end());
    }

    return {retScans, {retBins, retHashes}};
}

std::vector<int> TimsHashGenerator::hashSpectrum(MzSpectrumPL &spectrum) {

    int numRows = int(2000 * pow(10, this->resolution));

    auto sparseVec = toSparseVector(spectrum.vectorize(this->resolution), numRows);
    auto signumVec = calculateSignumVector(sparseVec, this->M, this->k, this->l);
    auto keys = calculateKeys(signumVec, 1, false);

    return keys;
}


std::map<int, std::set<int>> mergeBands(std::map<int, std::set<int>> b1, std::map<int, std::set<int>> b2){
    for(auto& [key, set]: b2){
        if(b1.contains(key)){
            b1[key].insert(set.begin(), set.end());
        }
        else {
            b1[key] = set;
        }
    }
    return b1;
}

std::pair<std::vector<int>, std::vector<int>> TimsHashGenerator::getCollisionInBands(Eigen::MatrixXi H, std::vector<int> scans, std::vector<int> bins) {

    auto singleCollisionInBox = [&scans, &bins](std::vector<int>& band) -> std::map<int, std::set<int>>{
        return collisionsInBox(band, scans, bins);
    };

    // take Hashing Matrix H and get all columns, effectively slicing it by bands
    std::vector<std::vector<int>> M;
    M.resize(H.cols());

    for(auto i = 0; i < H.cols(); i++){
        auto B = H.col(i);
        M[i].reserve(B.size());
        for(auto j = 0; j < B.size(); j++){
            M[i].push_back(B[j]);
        }
    }

    std::vector<std::map<int, std::set<int>>> retVec;
    retVec.resize(M.size());

    std::transform(std::execution::par_unseq, M.begin(), M.end(), retVec.begin(), singleCollisionInBox);

    std::map<int, std::set<int>> retMap {};

    for(std::map<int, std::set<int>> item: retVec){
        retMap = mergeBands(retMap, item);
    }

    std::vector<int> retBins, retScans;
    retBins.reserve(retMap.size());
    retScans.reserve(retMap.size());

    for(auto &[k, v]: retMap){
        retBins.insert(retBins.end(), v.size(), k);
        //retBins.push_back(k);
        retScans.insert(retScans.end(), v.begin(), v.end());
    }

    return {retBins, retScans};
}

#endif //SRC_HASHING_H
