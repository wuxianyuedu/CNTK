//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"

namespace CNTK
{
    class BasicTrainingSession : public TrainingSession
    {
    public:
        BasicTrainingSession(
            MinibatchSourcePtr trainingSource,
            Trainer& trainer,
            const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
            const MinibatchSizeSchedule& minibatchSizeSchedule,
            size_t checkpointFrequencyinSamples,
            const std::wstring& checkPointFileName);

        void Run(const DeviceDescriptor& computeDevice) override;

        void RestoreFromCheckpoint(const std::wstring& checkpointFileName) override;

    private:
        void SaveCheckpoint();

        MinibatchSourcePtr m_trainingSource;
        Trainer& m_trainer;
        std::unordered_map<Variable, StreamInformation> m_modelInputToMinibatchSourceStream;
        const MinibatchSizeSchedule m_minibatchSizeSchedule;
        const size_t m_checkpointFrequencyinSamples;
        const std::wstring m_checkPointFileName;
        size_t m_currentCheckpointIndex;

        BasicTrainingSession(const BasicTrainingSession&) = delete; BasicTrainingSession& operator=(const BasicTrainingSession&) = delete; BasicTrainingSession& operator=(BasicTrainingSession&&) = delete; BasicTrainingSession(BasicTrainingSession&& other) = delete;
    };
}