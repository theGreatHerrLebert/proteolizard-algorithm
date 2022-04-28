#include <vector>
#include <tuple>

#include "Hashing.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <ProteoLizardData/Spectrum.h>
#include <ProteoLizardData/Frame.h>

namespace py = pybind11;
PYBIND11_MODULE(libproteolizardalgorithm, h) {
    h.doc() = "proteolizard algorithm extension";
    py::class_<TimsHashGenerator>(h, "TimsHashGenerator")

            // -------------- CONSTRUCTOR ---------------
            .def(py::init<int, int, int, int, int>())

                    // -------------- MEMBER ---------------
            .def("getMatrixCopy", &TimsHashGenerator::getMatrixCopy)

            .def("hashMzSpectrum", [](TimsHashGenerator& self, MzSpectrumPL &spectrum){
                return py::array(py::cast(self.hashSpectrum(spectrum)));
            })
            .def("hashMzSpectrumWindows", [](
                    TimsHashGenerator &self,
                    MzSpectrumPL &spectrum,
                    int minPeaks,
                    int minIntensity,
                    double windowLength,
                    bool overlapping,
                    bool restricted) {

                auto p = self.hashSpectrum(spectrum, minPeaks,
                                           minIntensity, windowLength, overlapping, restricted);

                return py::make_tuple(py::array(py::cast(p.first)), py::array(py::cast(p.second)));
            })
            .def("hashTimsFrameWindows", [](TimsHashGenerator &self, TimsFramePL &frame, int minPeaks,
                                            int minIntensity,
                                            double windowLength,
                                            bool overlapping,
                                            bool restricted){

                auto p = self.hashFrame(frame, minPeaks, minIntensity, windowLength, overlapping, restricted);
                return py::make_tuple(py::array(py::cast(p.first)), py::array(py::cast(p.second.first)), py::array(py::cast(p.second.second)));
            })
            .def("calculateCollisions", [](TimsHashGenerator &self, py::array_t<int64_t> hashes,
                                           std::vector<int> scans, std::vector<int> bins){
                Eigen::MatrixXi H = hashes.cast<Eigen::MatrixXi>();
                auto p = self.getCollisionInBands(H, scans, bins);
                return py::make_tuple(py::array(py::cast(p.first)), py::array(py::cast(p.second)));
            })
            ;
}