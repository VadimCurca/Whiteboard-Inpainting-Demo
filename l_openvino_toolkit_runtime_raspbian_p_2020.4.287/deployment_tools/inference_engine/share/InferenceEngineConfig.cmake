# Copyright (C) 2018-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
#
# FindIE
# ------
#
#   You can specify the path to Inference Engine files in IE_ROOT_DIR
#
# This will define the following variables:
#
#   InferenceEngine_FOUND        - True if the system has the Inference Engine library
#   InferenceEngine_INCLUDE_DIRS - Inference Engine include directories
#   InferenceEngine_LIBRARIES    - Inference Engine libraries
#
# and the following imported targets:
#
#   IE::inference_engine            - The Inference Engine library
#   IE::inference_engine_legacy     - The Inference Engine library with legacy API for IR v7 and older.
#   IE::inference_engine_c_api      - The Inference Engine C API library
#

macro(ext_message TRACE_LEVEL)
    if (${TRACE_LEVEL} STREQUAL FATAL_ERROR)
        if(InferenceEngine_FIND_REQUIRED)
            message(FATAL_ERROR "${ARGN}")
        elseif(NOT InferenceEngine_FIND_QUIETLY)
            message(WARNING "${ARGN}")
        endif()
        return()
    elseif(NOT InferenceEngine_FIND_QUIETLY)
        message(${TRACE_LEVEL} "${ARGN}")
    endif ()
endmacro()

set(InferenceEngine_FOUND FALSE)

if(TARGET IE::inference_engine)
    set(InferenceEngine_FOUND TRUE)
    get_target_property(InferenceEngine_INCLUDE_DIRS IE::inference_engine INTERFACE_INCLUDE_DIRECTORIES)
    set(InferenceEngine_LIBRARIES IE::inference_engine_legacy IE::inference_engine
                                  IE::inference_engine_c_api)
else()
    if (WIN32)
        set(_ARCH intel64)
    else()
        string(TOLOWER ${CMAKE_SYSTEM_PROCESSOR} _ARCH)
        if(_ARCH STREQUAL "x86_64" OR _ARCH STREQUAL "amd64") # Windows detects Intel's 64-bit CPU as AMD64
            set(_ARCH intel64)
        elseif(_ARCH STREQUAL "i386")
            set(_ARCH ia32)
        endif()
    endif()

    set(THREADING "TBB")

    # check whether setvars.sh is sourced
    if(NOT IE_ROOT_DIR AND (DEFINED ENV{InferenceEngine_DIR} OR InferenceEngine_DIR OR DEFINED ENV{INTEL_OPENVINO_DIR}))
        if (EXISTS "${InferenceEngine_DIR}")
            # InferenceEngine_DIR manually set via command line params
            set(IE_ROOT_DIR "${InferenceEngine_DIR}/..")
        elseif (EXISTS "$ENV{InferenceEngine_DIR}")
            # InferenceEngine_DIR manually set via env
            set(IE_ROOT_DIR "$ENV{InferenceEngine_DIR}/..")
        elseif (EXISTS "$ENV{INTEL_OPENVINO_DIR}/inference_engine")
            # if we installed DL SDK
            set(IE_ROOT_DIR "$ENV{INTEL_OPENVINO_DIR}/inference_engine")
        elseif (EXISTS "$ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine")
            # CV SDK is installed
            set(IE_ROOT_DIR "$ENV{INTEL_OPENVINO_DIR}/deployment_tools/inference_engine")
        endif()
    endif()

    if(NOT IE_ROOT_DIR)
        ext_message(FATAL_ERROR "inference_engine root directory is not found")
    endif()

    find_path(IE_INCLUDE_DIR inference_engine.hpp "${IE_ROOT_DIR}/include" NO_DEFAULT_PATH)

    set(IE_LIB_DIR "${IE_ROOT_DIR}/lib/${_ARCH}")
    set(IE_LIB_REL_DIR "${IE_LIB_DIR}/Release")
    set(IE_LIB_DBG_DIR "${IE_LIB_DIR}/Debug")

    include(FindPackageHandleStandardArgs)

    if(WIN32)
        find_library(IE_RELEASE_LIBRARY inference_engine "${IE_LIB_REL_DIR}" NO_DEFAULT_PATH)
        find_library(IE_LEGACY_RELEASE_LIBRARY inference_engine_legacy "${IE_LIB_REL_DIR}" NO_DEFAULT_PATH)
        find_library(IE_C_API_RELEASE_LIBRARY inference_engine_c_api "${IE_LIB_REL_DIR}" NO_DEFAULT_PATH)
    elseif(APPLE)
        find_library(IE_RELEASE_LIBRARY inference_engine "${IE_LIB_DIR}" NO_DEFAULT_PATH)
        find_library(IE_LEGACY_RELEASE_LIBRARY inference_engine_legacy "${IE_LIB_DIR}" NO_DEFAULT_PATH)
        find_library(IE_C_API_RELEASE_LIBRARY inference_engine_c_api "${IE_LIB_DIR}" NO_DEFAULT_PATH)
    else()
        find_library(IE_RELEASE_LIBRARY inference_engine "${IE_LIB_DIR}" NO_DEFAULT_PATH)
        find_library(IE_LEGACY_RELEASE_LIBRARY inference_engine_legacy "${IE_LIB_DIR}" NO_DEFAULT_PATH)
        find_library(IE_C_API_RELEASE_LIBRARY inference_engine_c_api "${IE_LIB_DIR}" NO_DEFAULT_PATH)
    endif()

    find_package_handle_standard_args(  InferenceEngine
                                        FOUND_VAR INFERENCEENGINE_FOUND
                                        REQUIRED_VARS IE_RELEASE_LIBRARY IE_LEGACY_RELEASE_LIBRARY IE_C_API_RELEASE_LIBRARY IE_INCLUDE_DIR
                                        FAIL_MESSAGE "Some of mandatory Inference Engine components are not found. Please consult InferenceEgnineConfig.cmake module's help page.")

    if(INFERENCEENGINE_FOUND)
        # to keep this line for successful execution in CMake 2.8
        set(InferenceEngine_FOUND TRUE)

        foreach(ie_library_suffix "" "_legacy" "_c_api")
            string(TOUPPER "${ie_library_suffix}" ie_library_usuffix)
            add_library(IE::inference_engine${ie_library_suffix} SHARED IMPORTED GLOBAL)

            if (WIN32)
                set_target_properties(IE::inference_engine${ie_library_suffix} PROPERTIES
                        IMPORTED_CONFIGURATIONS RELEASE
                        IMPORTED_IMPLIB_RELEASE    "${IE${ie_library_usuffix}_RELEASE_LIBRARY}"
                        MAP_IMPORTED_CONFIG_RELEASE Release
                        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
                        INTERFACE_INCLUDE_DIRECTORIES "${IE_INCLUDE_DIR}")

                # Debug binaries are optional
                find_library(IE${ie_library_usuffix}_DEBUG_LIBRARY inference_engine${ie_library_suffix}d
                    "${IE_LIB_DBG_DIR}" NO_DEFAULT_PATH)
                if (IE${ie_library_usuffix}_DEBUG_LIBRARY)
                    set_property(TARGET IE::inference_engine${ie_library_suffix} APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
                    set_target_properties(IE::inference_engine${ie_library_suffix} PROPERTIES
                            IMPORTED_IMPLIB_DEBUG "${IE${ie_library_usuffix}_DEBUG_LIBRARY}"
                            MAP_IMPORTED_CONFIG_DEBUG Debug)
                else()
                    ext_message(WARNING "Inference Engine DEBUG binaries are missed.")
                endif()
            elseif (APPLE)
                set_target_properties(IE::inference_engine${ie_library_suffix} PROPERTIES
                        IMPORTED_LOCATION_RELEASE "${IE${ie_library_usuffix}_RELEASE_LIBRARY}"
                        INTERFACE_INCLUDE_DIRECTORIES "${IE_INCLUDE_DIR}"
                        INTERFACE_COMPILE_OPTIONS "-Wno-error=deprecated-declarations")

                # Debug binaries are optional
                find_library(IE${ie_library_usuffix}_DEBUG_LIBRARY inference_engine${ie_library_suffix}d "${IE_LIB_DIR}" NO_DEFAULT_PATH)
                if (IE${ie_library_usuffix}_DEBUG_LIBRARY)
                    set_target_properties(IE::inference_engine${ie_library_suffix} PROPERTIES
                            IMPORTED_LOCATION_DEBUG "${IE${ie_library_usuffix}_DEBUG_LIBRARY}")
                else()
                    ext_message(WARNING "Inference Engine DEBUG binaries are missed")
                endif()

                target_link_libraries(IE::inference_engine${ie_library_suffix} INTERFACE ${CMAKE_DL_LIBS})
            else()
                # Only Release binaries are distributed for Linux systems
                set_target_properties(IE::inference_engine${ie_library_suffix} PROPERTIES
                        IMPORTED_LOCATION "${IE${ie_library_usuffix}_RELEASE_LIBRARY}"
                        INTERFACE_INCLUDE_DIRECTORIES "${IE_INCLUDE_DIR}")
                if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
                    set_target_properties(IE::inference_engine${ie_library_suffix} PROPERTIES
                            INTERFACE_COMPILE_OPTIONS "-diag-warning=1786")
                else()
                    set_target_properties(IE::inference_engine${ie_library_suffix} PROPERTIES
                            INTERFACE_COMPILE_OPTIONS "-Wno-error=deprecated-declarations")
                endif()
                target_link_libraries(IE::inference_engine${ie_library_suffix} INTERFACE ${CMAKE_DL_LIBS})
            endif()
        endforeach()

        set(InferenceEngine_INCLUDE_DIRS ${IE_INCLUDE_DIR})
        set(InferenceEngine_LIBRARIES IE::inference_engine_legacy IE::inference_engine
                                      IE::inference_engine_c_api)

        set(IE_EXTERNAL_DIR "${IE_ROOT_DIR}/external")
        include("${IE_ROOT_DIR}/share/ie_parallel.cmake")
    endif()
endif()
