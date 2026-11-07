// Prints "<epoch seconds> <rssi dBm> <noise dBm> <tx rate Mbps>" four times a second.
import CoreWLAN
import Foundation
setvbuf(stdout, nil, _IOLBF, 0)
guard let wifi = CWWiFiClient.shared().interface() else { exit(1) }
while true {
    print(String(format: "%.3f %d %d %.0f", Date().timeIntervalSince1970, wifi.rssiValue(), wifi.noiseMeasurement(), wifi.transmitRate()))
    Thread.sleep(forTimeInterval: 0.25)
}
