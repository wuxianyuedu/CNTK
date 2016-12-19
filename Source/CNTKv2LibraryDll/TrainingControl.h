//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"

namespace CNTK
{
    class BasicTrainingControl : public TrainingControl
    {
    public:
        BasicTrainingControl(
            size_t minibatchSize,
            size_t maxTrainingSamplesCount,
            size_t checkpointFrequencyinSamples,
            const std::wstring& checkPointFileName);

        // Optional callback that gets called before each minbatch during training
        void PreMinibatchCallback(const Trainer& trainer) override;

        // Optional callback that gets called after each minbatch during training
        // Return value indicates whether the training should be stopped.
        bool PostMinibatchCallback(Trainer& trainer, bool minibatchTrainingResult) override;

        // Returns the desired size of the next minibatch
        size_t NextMinibatchSize() override;

    private:
        size_t m_minibatchSize;
        size_t m_maxTrainingSamplesCount;
        size_t m_checkpointFrequencyinSamples;
        std::wstring m_checkPointFileName;
        size_t m_currentCheckpointIndex;
    };
}