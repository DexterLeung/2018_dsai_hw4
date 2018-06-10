import gym
import numpy as np
import tensorflow as tf
import os
import random
import math
from datetime import datetime
import csv

random.seed()

INITIAL_LEARNING_RATES = [0.001, 0.0005, 0.00025];
BATCH_SIZES = [256, 128];
HIDDEN_SIZES = [100, 200, 100];
LOG_FREQ = 200;
WEIGHT_LOG_FREQ = 800;
REWARD_DEF = [0, 1, 2, 3];

def createVar(name, shape, initializer, l2loss = False, weightDecayRate = None, dtype=tf.float32):
    newVar = tf.get_variable(name, shape, initializer=initializer, dtype=dtype);
    if (l2loss):
        weightDecay = tf.multiply(tf.nn.l2_loss(newVar), weightDecayRate or 0, name="weightLoss");
        tf.add_to_collection("losses", weightDecay);
    return newVar;

class StateTransition:
    def __init__(self, step, fromState, toState, action, reward):
        self.step = step;
        self.fromState = fromState;
        self.toState = toState;
        self.action = action;
        self.reward = reward;

class DeepQLearning:
    ''' Class representing a deep-q learning.   --- UPDATED 20180605
    '''
    def __init__(self, inputItemShape, targetItemShape, trainName = "DQN", initialLearningRate = 0.005, batchSize = 32, layerHiddenSizes = [10, 10, 10], maxDataSteps = 1000, epsSoft = 0.8, gamma = 0.9, temporalUpdates = 1, delayedUpdateFreq = 50, logFreq = LOG_FREQ, weightLogFreq = WEIGHT_LOG_FREQ):
        self.trainName = trainName;
        folder = "D:/tmp/" + trainName + "/";
        nowDate = datetime.now();
        self.trainTime = '{:%m%d}'.format(nowDate) + " " + '{:%H%M}'.format(nowDate);
        self.folder = folder if folder is not None else "";
        print("New Folder: " + self.folder)
        os.makedirs(self.folder + "outputLogs/" + self.trainTime + "/", exist_ok=True);
        os.makedirs(self.folder + "tmpLogs/" + self.trainTime + "/", exist_ok=True);
        self.inputItemShape = inputItemShape;
        self.targetItemShape = targetItemShape;
        self.initialLearningRate = initialLearningRate;
        self.batchSize = batchSize;
        self.globalStep = self.localStep = 1;
        self.logFreq = logFreq;
        self.weightLogFreq = weightLogFreq;
        self.exponentialLossDecay = 0.9;
        self.exponentialVarDecay = 0.9999;
        self.optimizer = tf.train.AdamOptimizer;
        self.layerHiddenSizes = layerHiddenSizes;
        self.maxDataSteps = maxDataSteps;
        self.epsSoft = epsSoft;
        self.gamma = gamma;
        self.delayedUpdateFreq = delayedUpdateFreq;
        self.temporalUpdates = temporalUpdates;
        self.transitionReplacement = 0;
        self.bestMemories = [];
        self.bestRecords = [];
        self.curretMemory = [];
        self.useBestMemories = True;

        self._data = [];

        self.initialized = False;
        self._buildNo = 0;
        self.targetNetwork = self.evalNetwork = None;
        self.totalLossOP = self.avgLossOP = None;
        self._sess = None;
        self.__build__();

        self.timer = datetime.now();
        
    
    def getAction(self, toState, test = False):
        ''' Get the action using epsilon-soft policies.   --- UPDATED 20180605

            Parameters
            -----------------

            `toState`:      `list[float] `
            The state properties, converted into an ordered float values

            Returns
            -----------------

            `int` 
            The action id of the action to be taken
        '''
        
        if (self.initialized):
            # Set the epsilon
            epsilon = 0 if test else self.epsSoft;

            # If training is already initialized, run on the evaluation network to get the q-value of the observed new state.
            predictionInput = [toState];
            allQVals = self._sess.run(self.evalNetwork["qValue"], feed_dict={self.evalNetwork["inputTensor"]: predictionInput});

            # Find out the optimized action.
            bestActionID = np.argmax(allQVals);

            # Define the policy based on the extent of greedy/non-greedy approach, through a sorted list of probabilities whether the action can be chosen.
            actionProbs = [];
            for aid in range(0, self.targetItemShape):
                actionProbs.append(((1 - epsilon) if (aid == bestActionID) else 0) + epsilon/self.targetItemShape);
            
            # Get the action with the above policy
            tryProb = random.random();
            toActionID = 0;
            while tryProb > actionProbs[toActionID]:
                tryProb -= actionProbs[toActionID]
                toActionID += 1;
        
        else:
            # If training is not initialized, return a random action id.
            toActionID = math.floor(random.random() * self.targetItemShape);

        return toActionID;
        
    
    def addTransition(self, step, fromState, toState, action, reward):
        ''' Add a state transition observation to the training dataset.   --- UPDATED 20180605

        Parameters
        -----------------

        `step`:      `int`
        The step of the transition observation in an episode

        `fromState`:    `list[float] `
        The state properties, converted into ordered float values, before action taken

        `toState`:    `list[float] `
        The state properties, converted into ordered float values, after action taken

        `action`:   `int`
        The action taken to achieve the transition

        `reward`:   `float`
        The reward given by the action taken

        '''
        if (self.transitionReplacement == 0):
            # Add (or replace) transition to the training data set
            self._data.append(StateTransition(step, fromState, toState, action, reward));

            if (self.useBestMemories):
                self.curretMemory.append(StateTransition(step, fromState, toState, action, reward));

            # FIFO
            if len(self._data) > self.maxDataSteps:
                self._data = self._data[1:];
        
        elif (self.transitionReplacement == 1):
            # Replace the new state with low reard but high step
            if (len(self._data) <= self.maxDataSteps):
                self._data.append(StateTransition(step, fromState, toState, action, reward));
            else:
                memoryScore = [r.step/r.reward for r in self._data];
                maxScore = max(memoryScore);
                highest = [idx for idx,s in enumerate(memoryScore) if s == maxScore];
                if maxScore > (step/reward):
                    self._data[highest[0]] = StateTransition(step, fromState, toState, action, reward);
        
        elif (self.transitionReplacement == 2):
            # Add (or replace) transition to the training data set
            self._data.append(StateTransition(step, fromState, toState, action, reward));

            # FIFO
            if len(self._data) > self.maxDataSteps:
                meanReward = sum([r.reward for r in self._data]) / len(self._data);
                if (reward > meanReward):
                    lowRewardSteps = [(r.step if r.reward < meanReward else 0) for r in self._data];
                    maxStep = max(lowRewardSteps);
                    highest = [idx for idx,s in enumerate(lowRewardSteps) if s == maxStep];
                    del self._data[highest[0]];

    def updateBestMemory(self, step, push=True):
        if (self.useBestMemories):
            if (push):
                if (len(self.bestRecords) < 5) :
                    self.bestMemories.append(self.curretMemory);
                    self.bestRecords.append(step);
                elif (step < min(self.bestRecords)):
                    lowest = [idx for idx,s in enumerate(self.bestRecords) if s == max(self.bestRecords)];
                    del self.bestRecords[lowest[0]];
                    del self.bestMemories[lowest[0]];

                    self.bestMemories.append(self.curretMemory);
                    self.bestRecords.append(step);
            else:
                self.curretMemory = [];

    def buildNetwork(self):
        # Setup input and output tensors.
        inputTensor = tf.placeholder(tf.float32, [None, self.inputItemShape]);
        targetTensor = tf.placeholder(tf.float32, [None, self.targetItemShape]);
        weights = [];

        # Create DNN structure with given hidden sizes.
        prevLayer = inputTensor;
        for idx,hs in enumerate(self.layerHiddenSizes):
            w = createVar("w"+str(idx), [prevLayer.shape[-1], hs], tf.truncated_normal_initializer(mean=0,stddev=0.1, dtype=tf.float32), l2loss=False, weightDecayRate=0.05, dtype=tf.float32);
            b = createVar("b"+str(idx), [hs], tf.constant_initializer(0.001, dtype=tf.float32), l2loss=False,dtype=tf.float32);
            weights.append(w);
            weights.append(b);
            mid = tf.nn.bias_add(tf.matmul(prevLayer, w), b);
            mid = tf.layers.batch_normalization(mid, name="batchNorm" + str(idx));
            prevLayer = tf.nn.relu(mid, name="relu" + str(idx));
        
        # Create Regressor for the Q-values of different actions.
        w = createVar("RegressorWeight", [prevLayer.shape[-1], self.targetItemShape], tf.truncated_normal_initializer(mean=0,stddev=0.1, dtype=tf.float32), l2loss=False, weightDecayRate=0.05, dtype=tf.float32);
        b = createVar("RegressorBias", [self.targetItemShape], tf.constant_initializer(0.0, dtype=tf.float32), l2loss=False,dtype=tf.float32);
        weights.append(w);
        weights.append(b);
        qValue = tf.nn.bias_add(tf.matmul(prevLayer, w), b);

        # Initialize training steps.
        localStep = globalStep = self.localStep = self.globalStep = 1;
        self.localStepTensor = tf.Variable(self.localStep, trainable=False, dtype=tf.int64);
        self.globalStepTensor = tf.train.get_or_create_global_step();

        

        # Return the necessary tensors.
        return {"inputTensor": inputTensor, "targetTensor": targetTensor, "qValue": qValue, "weights": weights};
    
    def buildTargetNetwork(self):
        with self._graph.as_default():
            with tf.variable_scope("targetNetwork"):
                self.targetNetwork = self.buildNetwork();
        
    def buildEvaluationNetwork(self):
        with self._graph.as_default():
            with tf.variable_scope("evaluationNetwork"):
                self.evalNetwork = self.buildNetwork();
    
    def __build__(self):
        # Set up basic build initializations.
        self._buildNo += 1;
        self.localStep = 1;
        self.initialized = False;
        self.targetNetwork = self.evalNetwork = None;
        self.totalLossOP = self.avgLossOP = None;
        
        # Initialize graph and build the network.
        graph = self._graph = tf.Graph();
        self.buildTargetNetwork();
        self.buildEvaluationNetwork();

        # Start the session.
        self._sess = tf.InteractiveSession(graph=graph);

        # Initialize loggers.
        trainLogHeader = ["Timestamp", "Global Step", "Local Step", "Episode", "Total Loss", "Average Loss", "Latest Average Rewards", "Latest Total Rewards", "Latest Maximum Position", "Learning Rate", "Epsilon", "Examples per Batch", "Batch per Second"];
        self.trainLog = [trainLogHeader];

    def copyModelVars(self, scopeFrom, scopeTo):
        varsFrom = tf.get_collection(scopeFrom);
        varsTo = tf.get_collection(scopeTo);
        self._sess.run([tf.assign(varsTo[idx], valFrom) for idx, valFrom in enumerate(varsFrom)]);

    def close(self):
        self._sess.close();
        self._graph = None;

        with open(self.folder + "/outputLogs/" + self.trainTime + "/trainLog_" + str(self._buildNo) + ".csv", "w", encoding="utf-8", newline = "") as f:
            csvWriter = csv.writer(f);
            csvWriter.writerows(self.trainLog);
    
    def restart(self):
        self.close();
        self.__build__();

    def __trainInit__(self):
        with self._graph.as_default():
            # Finish the final loss by completing the loss function to minus the q value from the evaluation network.
            finalLoss = tf.reduce_mean(tf.squared_difference(self.evalNetwork["targetTensor"], self.evalNetwork["qValue"]));
            tf.add_to_collection("losses", finalLoss);

            # Accumulate all the losses and apply moving averages on it.
            totalLoss = tf.add_n(tf.get_collection("losses"), name='totalLoss');
            allLosses = tf.get_collection("losses");
            lossAvgs = tf.train.ExponentialMovingAverage(self.exponentialLossDecay, name='lossAvgs');
            lossAvgsOP = lossAvgs.apply(allLosses + [totalLoss]);

            # Creating the training operation.
            with tf.control_dependencies([lossAvgsOP]):
                opt = self.optimizer(self.initialLearningRate);
                applyGradientOP = opt.minimize(totalLoss, global_step=self.localStepTensor);
            assignStep = self.globalStepTensor.assign_add(1);

            # Take moving average decay on trainable variables.
            variableAvgs = tf.train.ExponentialMovingAverage(self.exponentialVarDecay, self.localStep);
            variableAvgsOP = variableAvgs.apply(tf.trainable_variables());

            # Grab the final training operation.
            with tf.control_dependencies([applyGradientOP, variableAvgsOP, assignStep]):
                self.totalLossOP = tf.add_n(allLosses, name='totalLoss');
                self.avgLossOP = self.totalLossOP/len(allLosses);
        
        self._sess.run(tf.global_variables_initializer());
        self.initialized = True;


    def __train__(self, newTrain = True, episode = 1):
        # Determine if this is a new build of training
        if ((newTrain or self.localStep == 1) and not self.initialized):
            self.__trainInit__();
            
        # For every temporal steps, train the network based on random batched of existing observations.
        finalStep = self.localStep + self.temporalUpdates;
        trainIdxs = [*range(0, self.maxDataSteps)];
        stepPerEpoch = self.maxDataSteps // self.batchSize;
        
        # Prepare nparray datasets
        if (self.useBestMemories):
            allData = [self._data, *self.bestMemories];
            fromStateData = np.concatenate([np.array([r.fromState for r in d]) for d in allData]);
            toStateData = np.concatenate([np.array([r.toState for r in d]) for d in allData]);
            actionData = np.concatenate([np.array([r.action for r in d]) for d in allData]);
            rewardData = np.concatenate([np.array([r.reward for r in d]) for d in allData]);
        else:
            fromStateData = np.array([r.fromState for r in self._data]);
            toStateData = np.array([r.toState for r in self._data]);
            actionData = np.array([r.action for r in self._data]);
            rewardData = np.array([r.reward for r in self._data]);
        
        for i in range(self.localStep, finalStep):
            # Update target network if needed.
            if (i % self.delayedUpdateFreq):
                self.copyModelVars("evaluationNetwork", "targetNetwork");

            # Shuffle indexes to be collected for training samples if needed.
            if (i % stepPerEpoch == 0):
                random.shuffle(trainIdxs);
            batchIdxs = trainIdxs[:self.batchSize];
            
            # Get the q values from the target network using the to-states. 
            targetInputData = toStateData[batchIdxs];
            targetQValue = self._sess.run(self.targetNetwork["qValue"], feed_dict={self.targetNetwork["inputTensor"]: targetInputData});

            # Evaluate the target data of the evaluation network:  r + \gamma max_a' Q(s',a';\theta^-_i).
            evalInputData = fromStateData[batchIdxs];
            evalTargetDataMaxA = rewardData[batchIdxs] + self.gamma * np.max(targetQValue, axis=1);

            # Noted the above values are on the q-values with max according to a', there are actually different q values on different a; so the loss will utilize the evaluation network info
            evalOriginalTargetData = self._sess.run(self.evalNetwork["qValue"], feed_dict={self.evalNetwork["inputTensor"]: evalInputData});

            # Update the original evaluation target data on max a', and the result is the evaluation target data.
            targetMaxA = actionData[batchIdxs];
            evalOriginalTargetData[[*range(0, self.batchSize)],targetMaxA] = evalTargetDataMaxA;
            evalTargetData = evalOriginalTargetData;

            # Train the evaluation network using the from-states.
            totalLossVal, avgLossVal = self._sess.run([self.totalLossOP, self.avgLossOP], feed_dict={self.evalNetwork["inputTensor"]: evalInputData, self.evalNetwork["targetTensor"]: evalTargetData});
            
            # Log training data if needed
            if (self.localStep % self.logFreq == 0):
                nowTime = datetime.now();
                duration = (nowTime - self.timer).total_seconds();
                self.timer = nowTime;
                examplesPerSec = self.logFreq * self.batchSize / duration;
                secPerBatch = float(duration / self.logFreq);
                logLR = self.initialLearningRate;
                oriRewards = np.array([r.reward for r in self._data]) if self.useBestMemories else rewardData;
                maxRewards = np.max(oriRewards);
                avgRewards = np.sum(oriRewards)/len(oriRewards);
                maxPosition = max(toStateData[:,0]);
                self.trainLog.append([nowTime, self.globalStep, i, episode, totalLossVal, avgLossVal, avgRewards, maxRewards, maxPosition, logLR, self.epsSoft, examplesPerSec, secPerBatch]);
                print('%s: Episode %d, Step %d (%d) --- loss: %f; max reward: %f, max position: %f, learning rate: %f; epsilon: %f;  %.1f examples/s; %.3f s/batch' % (nowTime, episode, self.globalStep, i, avgLossVal, maxRewards, maxPosition, logLR, self.epsSoft, examplesPerSec, secPerBatch));

            if (self.epsSoft > 0.1):
                self.epsSoft -= 0.0001;
            self.localStep += 1;
            self.globalStep += 1;
    
    def continueTrain(self, episode):
        # Initialize the local step and global step
        self.__train__(newTrain = False, episode = episode);
            


def dqnMountainCar(trainName, maxDataSteps, totalEpoches, buildCount, rewardDefinition, learningRate, batchSize, hiddenSizes):
    # Open the MountainCar-v0 
    env = gym.make('MountainCar-v0');

    # Remove default timestep boundary at 200.
    env = env.unwrapped;

    mountainCar = DeepQLearning(env.observation_space.shape[0], env.action_space.n, trainName = trainName, initialLearningRate = learningRate, batchSize = batchSize, layerHiddenSizes = hiddenSizes, maxDataSteps = maxDataSteps);
    
    
    
    episodeRecordsHeader = ["Episode", "Finishes at", "Latest Average Rewards", "Latest Maximum Rewards", "Latest Maximum Position"];
    for b in range(0, buildCount):
        if (b>0):
            mountainCar.restart();
        episodeRecord = [episodeRecordsHeader];

        learningState = 0;
        maxPos = env.observation_space.low[0];
        trialMaxSteps = 2*maxDataSteps;

        for i_episode in range(totalEpoches):
            observation = env.reset();
            print("Episode: ", i_episode);

            if (i_episode == totalEpoches - 1):
                learningState = 2;
            
            if (learningState == 2):
                trialMaxSteps = 3*maxDataSteps;
            
            for t in range(0, trialMaxSteps):
                # 1. Render environment based on the observation
                if (learningState == 2):
                    env.render();

                # 2. Choose Actions from previous states
                action = mountainCar.getAction(observation, test = learningState == 2);

                # 3. Get the next observation from the action done
                nextObservation, reward, done, info = env.step(action);

                # 4. Compute a customized reward  (Original environment rewards is always -1)
                if (rewardDefinition == 1):
                    # Reward depending on the height 
                    cuzRewards = nextObservation[0] - env.observation_space.low[0];
                elif (rewardDefinition == 2):
                    # Reward depending on whether done or not
                    cuzRewards = (10*maxDataSteps) if done else reward;
                elif (rewardDefinition == 3):
                    # Reward depending on the height 
                    maxPos = max(maxPos, nextObservation[0]);
                    cuzRewards = maxPos**2 / (i_episode+0.001);
                else:
                    # Constant reward as default -1
                    cuzRewards = reward;

                # 5. Append a record of state transition with the action done and resulted rewards
                if (learningState != 2):
                    mountainCar.addTransition(t, observation, nextObservation, action, cuzRewards);

                # 6. Train the model after collecting enough state transitions
                if (learningState == 1):
                    mountainCar.continueTrain(i_episode);
                elif (learningState == 0 and t == maxDataSteps - 1):
                    learningState = 1;

                # 7. Log Episode information
                if done:
                    print("Episode finished after {} timesteps".format(t+1));
                    mountainCar.updateBestMemory(t+1, push=True);
                    finalRewards = [r.reward for r in mountainCar._data];
                    finalPositions = [r.toState[0] for r in mountainCar._data];
                    episodeRecord.append([i_episode, t+1, sum(finalRewards)/len(finalRewards), max(finalRewards), max(finalPositions)]);
                    break;
                elif t == (trialMaxSteps - 1):
                    mountainCar.updateBestMemory(t+1, push=False);
                    finalRewards = [r.reward for r in mountainCar._data];
                    finalPositions = [r.toState[0] for r in mountainCar._data];
                    episodeRecord.append([i_episode, "Failed", sum(finalRewards)/len(finalRewards), max(finalRewards), max(finalPositions)]);
        
        with open(mountainCar.folder + "/outputLogs/" + mountainCar.trainTime + "/episodeLog_" + str(b+1) + ".csv", "w", encoding="utf-8", newline = "") as f:
            csvWriter = csv.writer(f);
            csvWriter.writerows(episodeRecord);
        episodeRecord = [episodeRecordsHeader];
    
    print("Test");

    env.close();
    mountainCar.close();
    

dqnMountainCar("DQN-1", 400, 250, 5, REWARD_DEF[1], INITIAL_LEARNING_RATES[0], BATCH_SIZES[0], HIDDEN_SIZES);
dqnMountainCar("DQN-2", 400, 250, 5, REWARD_DEF[1], INITIAL_LEARNING_RATES[1], BATCH_SIZES[0], HIDDEN_SIZES);
dqnMountainCar("DQN-3", 400, 250, 5, REWARD_DEF[1], INITIAL_LEARNING_RATES[0], BATCH_SIZES[1], HIDDEN_SIZES);
dqnMountainCar("DQN-4", 800, 400, 5, REWARD_DEF[0], INITIAL_LEARNING_RATES[0], BATCH_SIZES[0], HIDDEN_SIZES);
dqnMountainCar("DQN-5", 800, 400, 5, REWARD_DEF[2], INITIAL_LEARNING_RATES[0], BATCH_SIZES[0], HIDDEN_SIZES);