
import constants
import numpy as np
from matplotlib import pyplot as plt


class SimplePendulum:
    def __init__(self):
        self.A = np.array([[np.pi, constants.deltaT], 
                           [-constants.g/constants.l, -constants.b/(constants.m*constants.l**2)]])
        self.b = np.array([[0.0], [1.0/(constants.m*constants.l**2)]])
    
    def groundTruth(self, x_prior, u):
        self.A = np.array([[x_prior[0], constants.deltaT], 
                    [-constants.g/constants.l, -constants.b/(constants.m*constants.l**2)]])
        return np.matmul(self.A, x_prior) + np.matmul(self.b, u)

    def predictor(self, x_prior, u):
        noise = 0.3*np.random.random(2)
        self.A = np.array([[x_prior[0], constants.deltaT], 
            [-constants.g/constants.l, -constants.b/(constants.m*constants.l**2)]])
 
        x_predictor = np.matmul(self.A,x_prior) + np.matmul(self.b, u)
        x_predictor += noise

        return x_predictor
        

class MeasurementSimulator(SimplePendulum):
    def __init__(self):
        SimplePendulum.__init__(self)   #if this __init__ method is not explicitly added, the __init__ for measurementSimulator overwrites it
        varx1 = 0.3  #variance on position (from encoder)
        varx2 = 0.3 
        self.R = np.array([[varx1, 0.0], 
                          [0.0, varx2]])      #assumes the velocity has its own sensor and its uncorrelated to position

    def sensorOutput(self, x_prior, u, std=0.7):
        x_actual = self.groundTruth(x_prior, u)
        return x_actual + std*np.random.random(2)

class kalmanFilter():
    def __init__(self):
        self.SigmaApriori = 0.1*np.eye(2)
        self.SigmaPrediction = np.eye(2)
        self.pendulum = SimplePendulum()
        self.sensor = MeasurementSimulator()

    def posterior(self, x_prior, u=np.array([0])):
        self.pendulum.A = np.array([[x_prior[0], constants.deltaT], 
            [-constants.g/constants.l, -constants.b/(constants.m*constants.l**2)]])
 
        self.SigmaApriori = np.matmul(self.pendulum.A, self.SigmaPrediction, np.transpose(self.pendulum.A)) #Qt = 0

        z = self.sensor.sensorOutput(x_prior, u)
        x_apriori = self.pendulum.predictor(x_prior, u)
        K = np.matmul(self.SigmaApriori, np.linalg.inv(self.SigmaApriori + self.sensor.R))
        self.SigmaPrediction = np.matmul((np.eye(2) - K), self.SigmaApriori)

        return x_apriori + np.matmul(K, (z - x_apriori))


def integrate():

    kf = kalmanFilter()

    x0 = np.array([np.pi, 0.0])
    X = []
    X.append(x0)

    for t in np.arange(0, 1, constants.deltaT):
        x = kf.posterior(X[-1])
        x = (x + np.pi) % (2 * np.pi) - np.pi
        X.append(x)
        # print(X)

    return X

if __name__=="__main__":
    X = integrate()
    theta = [x[0] for x in X]
    thetadot = [x[1] for x in X]
    plt.plot(theta,thetadot, color="black")
    plt.show()