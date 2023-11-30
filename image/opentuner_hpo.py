import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter, EnumParameter, FloatParameter
from opentuner import MeasurementInterface
from opentuner import Result
from common import hpo_space, eval


scores = []
class MyTuner(MeasurementInterface):
    def manipulator(self):
        manipulator = ConfigurationManipulator()
        
        for key, val in hpo_space.items():
            if isinstance(val, list):
                manipulator.add_parameter(
                    EnumParameter(key, val)
                )
            elif isinstance(val, tuple):
                if isinstance(val[0], int):
                    manipulator.add_parameter(
                        IntegerParameter(key, val[0], val[1])
                    )
                else:
                    manipulator.add_parameter(
                        FloatParameter(key, val[0], val[1])
                    )

        return manipulator
    
    def run(self, desired_result, input, limit):
        cfg = desired_result.configuration.data

        try:
            score = eval(cfg)
        except:
            score = 0.

        scores.append(score)
        print('[Accuracy]', score)

        return Result(time=100.-score)


if __name__ == "__main__":
    argparser = opentuner.default_argparser()
    MyTuner.main(argparser.parse_args())
    print('Scores:', scores)
    print('Score:', max(scores))

