class KerasPilot(ABC):
    """
    Base class for Keras models that will provide steering and throttle to
    guide a car.
    """
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3)) -> None:
        # self.model: Optional[Model] = None
        self.input_shape = input_shape
        self.optimizer = "adam"
        self.interpreter = interpreter
        self.interpreter.set_model(self)
        logger.info(f'Created {self} with interpreter: {interpreter}')

    def load(self, model_path: str) -> None:
        logger.info(f'Loading model {model_path}')
        self.interpreter.load(model_path)

    def load_weights(self, model_path: str, by_name: bool = True) -> None:
        self.interpreter.load_weights(model_path, by_name=by_name)

    def shutdown(self) -> None:
        pass

    def compile(self) -> None:
        pass

    @abstractmethod
    def create_model(self):
        pass

    def set_optimizer(self, optimizer_type: str,
                      rate: float, decay: float) -> None:
        if optimizer_type == "adam":
            optimizer = keras.optimizers.Adam(lr=rate, decay=decay)
        elif optimizer_type == "sgd":
            optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay)
        else:
            raise Exception(f"Unknown optimizer type: {optimizer_type}")
        self.interpreter.set_optimizer(optimizer)

    def get_input_shapes(self) -> List[tf.TensorShape]:
        return self.interpreter.get_input_shapes()

    def seq_size(self) -> int:
        return 0

    def run(self, img_arr: np.ndarray, other_arr: List[float] = None) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """
        Donkeycar parts interface to run the part in the loop.
        :param img_arr:     uint8 [0,255] numpy array with image data
        :param other_arr:   numpy array of additional data to be used in the
                            pilot, like IMU array for the IMU model or a
                            state vector in the Behavioural model
        :return:            tuple of (angle, throttle)
        """
        norm_arr = normalize_image(img_arr)
        np_other_array = np.array(other_arr) if other_arr else None
        return self.inference(norm_arr, np_other_array)

    def inference(self, img_arr: np.ndarray, other_arr: Optional[np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ Inferencing using the interpreter
            :param img_arr:     float32 [0,1] numpy array with normalized image
                                data
            :param other_arr:   numpy array of additional data to be used in the
                                pilot, like IMU array for the IMU model or a
                                state vector in the Behavioural model
            :return:            tuple of (angle, throttle)
        """
        out = self.interpreter.predict(img_arr, other_arr)
        return self.interpreter_to_output(out)

    def inference_from_dict(self, input_dict: Dict[str, np.ndarray]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ Inferencing using the interpreter
            :param input_dict:  input dictionary of str and np.ndarray
            :return:            typically tuple of (angle, throttle)
        """
        output = self.interpreter.predict_from_dict(input_dict)
        return self.interpreter_to_output(output)

    @abstractmethod
    def interpreter_to_output(
            self,
            interpreter_out: Sequence[Union[float, np.ndarray]]) \
            -> Tuple[Union[float, np.ndarray], ...]:
        """ Virtual method to be implemented by child classes for conversion
            :param interpreter_out:  input data
            :return:                 output values, possibly tuple of np.ndarray
        """
        pass

    def train(self,
              model_path: str,
              train_data: Union[DatasetV1, DatasetV2],
              train_steps: int,
              batch_size: int,
              validation_data: Union[DatasetV1, DatasetV2],
              validation_steps: int,
              epochs: int,
              verbose: int = 1,
              min_delta: float = .0005,
              patience: int = 5,
              show_plot: bool = False) -> tf.keras.callbacks.History:
        """
        trains the model
        """
        log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
        
        assert isinstance(self.interpreter, KerasInterpreter)
        model = self.interpreter.model
        self.compile()

        callbacks = [
            EarlyStopping(monitor='val_loss',
                          patience=patience,
                          min_delta=min_delta),
            ModelCheckpoint(monitor='val_loss',
                            filepath=model_path,
                            save_best_only=True,
                            verbose=verbose)]
        
        

        history: tf.keras.callbacks.History = model.fit(
            x=train_data,
            steps_per_epoch=train_steps,
            batch_size=batch_size,
            callbacks=callbacks,
            validation_data=validation_data,
            validation_steps=validation_steps,
            epochs=epochs,
            verbose=verbose,
            workers=1,
            use_multiprocessing=False)

        if show_plot:
            try:
                import matplotlib.pyplot as plt
                from pathlib import Path

                plt.figure(1)
                # Only do accuracy if we have that data
                # (e.g. categorical outputs)
                if 'angle_out_acc' in history.history:
                    plt.subplot(121)

                # summarize history for loss
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('model loss')
                plt.ylabel('loss')
                plt.xlabel('epoch')
                plt.legend(['train', 'validate'], loc='upper right')

                # summarize history for acc
                if 'angle_out_acc' in history.history:
                    plt.subplot(122)
                    plt.plot(history.history['angle_out_acc'])
                    plt.plot(history.history['val_angle_out_acc'])
                    plt.title('model angle accuracy')
                    plt.ylabel('acc')
                    plt.xlabel('epoch')

                plt.savefig(Path(model_path).with_suffix('.png'))
                # plt.show()

            except Exception as ex:
                print(f"problems with loss graph: {ex}")

        return history.history

    def x_transform(
            self,
            record: Union[TubRecord, List[TubRecord]],
            img_processor: Callable[[np.ndarray], np.ndarray]) \
            -> Dict[str, Union[float, np.ndarray]]:
        """ Transforms the record into dictionary for x for training the
        model to x,y, and applies an image augmentation. Here we assume the
        model only takes the image as input. All model input layer's names
        must be matched by dictionary keys."""
        assert isinstance(record, TubRecord), "TubRecord required"
        img_arr = record.image(processor=img_processor)
        return {'img_in': img_arr}

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        """ Transforms the record into dictionary for y for training the
        model to x,y. All model ouputs layer's names must be matched by
        dictionary keys. """
        raise NotImplementedError(f'{self} not ready yet for new training '
                                  f'pipeline')

    def output_types(self) -> Tuple[Dict[str, np.typename], ...]:
        """ Used in tf.data, assume all types are doubles"""
        shapes = self.output_shapes()
        types = tuple({k: tf.float64 for k in d} for d in shapes)
        return types

    def output_shapes(self) -> Dict[str, tf.TensorShape]:
        return {}

    def __str__(self) -> str:
        """ For printing model initialisation """
        return type(self).__name__



class KerasLinear(KerasPilot):
    
    #The KerasLinear pilot uses one neuron to output a continuous value via
    #the Keras Dense layer with linear activation. One each for steering and
    #throttle. The output is not bounded.
    
    def __init__(self,
                 interpreter: Interpreter = KerasInterpreter(),
                 input_shape: Tuple[int, ...] = (120, 160, 3),
                 num_outputs: int = 2):
        self.num_outputs = num_outputs
        super().__init__(interpreter, input_shape)

    def create_model(self):
        return default_n_linear(self.num_outputs, self.input_shape)

    def compile(self):
        #self.interpreter.compile(optimizer=self.optimizer, loss='mse')
        self.interpreter.compile(optimizer="adam", loss='AHLR')

    def interpreter_to_output(self, interpreter_out):
        steering = interpreter_out[0]
        throttle = interpreter_out[1]
        return steering[0], throttle[0]

    def y_transform(self, record: Union[TubRecord, List[TubRecord]]) \
            -> Dict[str, Union[float, List[float]]]:
        assert isinstance(record, TubRecord), 'TubRecord expected'
        angle: float = record.underlying['user/angle']
        throttle: float = record.underlying['user/throttle']
        return {'n_outputs0': angle, 'n_outputs1': throttle}

    def output_shapes(self):
        # need to cut off None from [None, 120, 160, 3] tensor shape
        img_shape = self.get_input_shapes()[0][1:]
        shapes = ({'img_in': tf.TensorShape(img_shape)},
                  {'n_outputs0': tf.TensorShape([]),
                   'n_outputs1': tf.TensorShape([])})
        return shapes




def conv2d(filters, kernel, strides, layer_num, activation='relu'):
   
    return Convolution2D(filters=filters,
                         kernel_size=(kernel, kernel),
                         strides=(strides, strides),
                         activation=activation,
                         name='conv2d_' + str(layer_num))
  
def core_cnn_layers(img_in, drop, l4_stride=1):
    x = img_in
    x = conv2d(32, 3, 2, 1)(x)
    x = Dropout(drop)(x)
    x = conv2d(64, 3, 2, 2)(x)
    x = Dropout(drop)(x)
    x = conv2d(128, 3, 2, 3)(x)
    x = Dropout(drop)(x)
    x = conv2d(128, 3, l4_stride, 4)(x)
    x = Dropout(drop)(x)
    x = Flatten(name='flattened')(x)
    return x

def default_n_linear(num_outputs, input_shape=(120, 160, 3)):
    drop = 0.2
    img_in = Input(shape=input_shape, name='img_in')
    x = core_cnn_layers(img_in, drop)
    x = Dense(256, activation='relu', name='dense_1')(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu', name='dense_2')(x)
    x = Dropout(drop)(x)

    outputs = []
    for i in range(num_outputs):
        outputs.append(
            Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs, name='linear')
    return model
