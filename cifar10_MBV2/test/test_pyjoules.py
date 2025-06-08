from pyJoules.energy_meter import EnergyMeter
from pyJoules.device import DeviceFactory
from pyJoules.device.rapl_device import RaplPackageDomain
from pyJoules.handler import PrintHandler

# 你要测量的代码块
def target():
    x = 0
    for i in range(10_000_000):
        x += i ** 0.5
    return x

# 设置测量 CPU 0 的 RAPL 域
domains = [RaplPackageDomain(0)]
devices = DeviceFactory.create_devices(domains)
meter = EnergyMeter(devices)

# 开始测量
meter.start(tag='run')
target()
meter.stop()

# 输出结果（焦耳）
meter.record(PrintHandler())
