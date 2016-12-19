//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "TrainingControl.h"

namespace CNTK
{
    TrainingControlPtr BasicTrainingControl(size_t maxTrainingSamplesCount, size_t checkpointFrequencyinSamples, const std::pair<std::wstring, std::wstring>& modelAndCheckpointFileNames);

    BasicTrainingControl::BasicTrainingControl(size_t minibatchSize, size_t maxTrainingSamplesCount, size_t checkpointFrequencyinSamples, const std::wstring checkPointFileName) :
        m_minibatchSize(minibatchSize),
        m_maxTrainingSamplesCount(maxTrainingSamplesCount),
        m_checkpointFrequencyinSamples(checkpointFrequencyinSamples),
        m_checkPointFileName(checkPointFileName),
        m_currentCheckpointIndex(0)
    {
    }

    // Optional callback that gets called before each minbatch during training
    void BasicTrainingControl::PreMinibatchCallback(const Trainer&)
    {
    }

    // Optional callback that gets called after each minbatch during training
    // Return value indicates whether the training should be stopped.
    bool BasicTrainingControl::PostMinibatchCallback(Trainer& trainer, bool minibatchTrainingResult)
    {
        size_t checkpointIndex = trainer.TotalNumberOfSamplesSeen() / m_checkpointFrequencyinSamples;
        if (checkpointIndex > m_currentCheckpointIndex || minibatchTrainingResult)
        {
            // What to do with external state?
            trainer.SaveCheckpoint(m_checkPointFileName);
            m_currentCheckpointIndex = checkpointIndex;
        }

        return minibatchTrainingResult && trainer.TotalNumberOfSamplesSeen() < m_maxTrainingSamplesCount;
    }

    // Returns the desired size of the next minibatch
    size_t BasicTrainingControl::NextMinibatchSize()
    {
        return m_minibatchSize;
    }
}
