#pragma once
#include "opencl_init.h"
#include "CL/opencl.hpp"

namespace cl_utils {
    class KernelBuilder {
    private:
        cl::Kernel kernel;
        int arg_counter;
    public:
        KernelBuilder(const cl::Program& program, const char* name);

        /*! \brief setArg overload taking a POD type */
        template <typename... T>
        KernelBuilder& args(const T&... values)
        {
            (kernel.setArg<T>(arg_counter++, values), ...);
            return *this;
        }

        /*! \brief build the cl::Kernel object */
        cl::Kernel build() const;
    };

    //helper method to copy an OpenCL buffer into an OpenCL Image (fast copy that happens in the device)
    cl::Image2D copyBufferToImage(const cl::Context& context, const cl::CommandQueue& queue, const cl_mem* image_buff, const long long rows, const  long long cols);
}