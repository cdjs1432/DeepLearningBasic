from Layers import *
from ch3_ConvNet.ConvLayers import ConvLayer


class Model:
    def __init__(self):
        self.params = {}
        self.grads = {}
        self.keys = []
        self.layers = {}
        self.num = 0
        self.l2 = {}
        self.size = {}
        self.weight_decay_lambda = {}
        self.loss = None
        self.pred = None

    def addlayer(self, layer, name=None):
        if name is None:
            name = str(self.num)

        self.keys.append(name)
        self.num += 1
        self.layers[name] = layer

    def forward(self, x, train_flag=True):
        if x.shape[0] > 1024:  # if too much input size: memory :(
            temp_x = np.zeros((x.shape[0], 10))
            for i in range(0, x.shape[0], x.shape[0] // 10):
                temp_x[i:i + x.shape[0] // 10] = self.forward(x[i:i + x.shape[0] // 10], False)
            x = temp_x
        else:
            for i in range(len(self.keys) - 1):
                key = self.keys[i]
                if isinstance(self.layers[key], Dropout):
                    x = self.layers[key].forward(x, train_flag)
                else:
                    x = self.layers[key].forward(x)
                if key in self.weight_decay_lambda:
                    self.l2[key] = np.sum(np.square(self.params[key])) * self.weight_decay_lambda[key]
        return x

    def predict(self, x):
        x = self.forward(x, False)
        self.pred = softmax(x)
        return self.pred

    def eval(self, x, y, epoch=None):
        x = self.forward(x, False)
        self.loss = self.layers[self.keys[-1]].forward(x, y)
        self.loss += sum(self.l2.values()) / 2
        self.pred = softmax(x)

        if epoch is None:
            print("ACC : ", (self.pred.argmax(1) == y.argmax(1)).mean())
            print("LOSS : ", self.loss)
        else:
            print("ACC on epoch %d : " % epoch, (self.pred.argmax(1) == y.argmax(1)).mean())
            print("LOSS on epoch %d : " % epoch, self.loss)

    def train(self, x_train, y_train, optimizer, epoch, learning_rate, skip_init=False):
        if not skip_init:
            in_size = x_train.shape[1:]
            for name in self.layers:
                if not self.layers[name].activation:
                    out_size = (self.layers[name].out,)
                    if isinstance(self.layers[name], ConvLayer):
                        out_size = out_size[0]
                        fn, fh, fw = out_size
                        c, h, w = in_size
                        size = (fn, c, fh, fw)
                        out_h = int(1 + (h + 2 * self.layers[name].pad - fh) / self.layers[name].stride)
                        out_w = int(1 + (w + 2 * self.layers[name].pad - fw) / self.layers[name].stride)
                        in_size = (fn, out_h, out_w)
                    else:
                        size = (*in_size, *out_size)
                        in_size = (*out_size,)
                    self.size[name] = size

                    self.weight_decay_lambda[name] = self.layers[name].reg
                    if isinstance(self.layers[name], AddLayer):
                        self.params[name] = np.zeros(self.layers[name].out)
                    elif self.layers[name].init is 'xavier':
                        self.params[name] = xavier_initialization(size)
                    elif self.layers[name].init is 'he':
                        self.params[name] = he_initialization(size)
                    else:
                        self.params[name] = np.random.uniform(size)

                    self.layers[name].param = self.params[name]
                elif isinstance(self.layers[name], Flatten):
                    self.layers[name].shape = (fn, out_h, out_w)
                    tmp_size = 1
                    for items in in_size:
                        tmp_size *= items
                    in_size = (tmp_size,)
        optimizer.train(x_train, y_train, epoch, learning_rate, self)

    def save(self, path=''):  # Model save
        f = open(path + "model.txt", 'w')  # model.txt: 모델의 구조 저장
        # 파일에 가장 앞에 가중치를 저장할 weight.npz 파일의 경로를 저장함
        f.write(path + "weight.npz\n")  # weight.npz: Layer의 가중치값 저장

        params = {}  # np.savez를 위해 가중치들을 저장할 변수
        for name in self.layers:  # layers의 이름들을 가져와서:
            # 해당 값들의 클래스명과 이름을 불러옴
            data = self.layers[name].__class__.__name__ + "\n" + name + "\n"
            if not self.layers[name].activation:  # 가중치를 갖는 layer라면:
                params[name] = self.layers[name].param  # 가중치 저장
            f.write(data)  # 데이터를 model.txt에 저장함
        np.savez(path + "weight", **params)  # 가중치값들을 weight.npz에 저장

    def load(self, path):  # Model load
        f = open(path)  # 입력된 경로의 파일
        weight_path = f.readline()[:-1]  # 가장 먼저 들어있는 weight.npz의 경로 가져오기
        load = np.load(weight_path)  # 가중치 load 변수에 불러오기
        while True:
            layer = f.readline()[:-1]  # layer 종류 불러오기
            if not layer:  # 없으면 파일이 끝난 것이므로 종료
                break
            name = f.readline()[:-1]  # name 불러오기

            self.addlayer(eval(layer)(), name)  # layer 클래스에 name을 넣어서 생성
            # Model.addlayer은 인공지능 모델에 layer를 넣는 함수임
            if name in load:  # 만약 불러온 가중치 변수에 해당하는 name이라면:
                self.layers[name].param = load[name]  # 가중치 변수값 저장
