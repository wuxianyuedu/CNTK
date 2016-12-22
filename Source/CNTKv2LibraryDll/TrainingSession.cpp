//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "TrainingSession.h"
#include "fileutil.h"

namespace
{
    const std::wstring g_checkpointIndex = L"CheckpointIndex";
    const std::wstring g_trainingMinibatchSource = L"TrainingMinibatchSource";
}

namespace CNTK
{
    TrainingSessionPtr CreateBasicTrainingSession(MinibatchSourcePtr trainingSource,
        Trainer& trainer,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        size_t checkpointFrequencyinSamples,
        const std::wstring& checkPointFileName)
    {
        return MakeSharedObject<BasicTrainingSession>(trainingSource,
            trainer,
            modelInputToMinibatchSourceStream,
            minibatchSizeSchedule,
            checkpointFrequencyinSamples,
            checkPointFileName);
    }

    BasicTrainingSession::BasicTrainingSession(
        MinibatchSourcePtr trainingSource,
        Trainer& trainer,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        size_t checkpointFrequencyInSamples,
        const std::wstring& checkPointFileName) :
        m_trainingSource(trainingSource),
        m_trainer(trainer),
        m_modelInputToMinibatchSourceStream(modelInputToMinibatchSourceStream),
        m_minibatchSizeSchedule(minibatchSizeSchedule),
        m_checkpointFrequencyinSamples(checkpointFrequencyInSamples),
        m_checkPointFileName(checkPointFileName),
        m_currentCheckpointIndex(0)
    {
        if (m_minibatchSizeSchedule.Unit() == MinibatchSizeSchedule::UnitType::Minibatch)
            RuntimeError("Currently CNTK only support minibatch size schedule based on samples.");
    }

    void BasicTrainingSession::Run()
    {
        std::unordered_map<Variable, ValuePtr> minibatch;
        bool shouldTrain = true;
        while (shouldTrain)
        {
            size_t mbSize = m_minibatchSizeSchedule[m_trainer.TotalNumberOfSamplesSeen()];
            auto minibatchData = m_trainingSource->GetNextMinibatch(mbSize);

            minibatch.clear();
            if (!minibatchData.empty())
            {
                for (auto v : m_modelInputToMinibatchSourceStream)
                    minibatch.insert({ v.first, minibatchData[v.second].m_data });
            }

            shouldTrain = m_trainer.TrainMinibatch(minibatch);

            // Check whether to create a checkpoint
            size_t checkpointIndex = m_trainer.TotalNumberOfSamplesSeen() / m_checkpointFrequencyinSamples;
            if (checkpointIndex > m_currentCheckpointIndex)
            {
                m_currentCheckpointIndex = checkpointIndex;
                SaveCheckpoint();
            }
        }

        SaveCheckpoint();
    }

    void BasicTrainingSession::RestoreFromCheckpoint(const std::wstring& checkpointFileName)
    {
        Dictionary externalState = m_trainer.RestoreFromCheckpoint(checkpointFileName);
        m_currentCheckpointIndex = externalState[g_checkpointIndex].Value<size_t>();
        m_trainingSource->RestoreFromCheckpoint(externalState[g_trainingMinibatchSource].Value<Dictionary>());
    }

    void BasicTrainingSession::SaveCheckpoint()
    {
        Dictionary externalState;
        externalState[g_checkpointIndex] = m_currentCheckpointIndex;
        externalState[g_trainingMinibatchSource] = m_trainingSource->GetCheckpointState();

        std::wstring tempFileName = m_checkPointFileName + L".tmp";
        m_trainer.SaveCheckpoint(m_checkPointFileName, externalState);

        _wunlink(m_checkPointFileName.c_str());
        renameOrDie(tempFileName, m_checkPointFileName);
    }
}
