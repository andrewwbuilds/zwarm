// Joins an open drone network by name. networksetup can answer "Could not find network" for one a scan has just seen.
import CoreWLAN
import Foundation

guard CommandLine.arguments.count == 2, let wifi = CWWiFiClient.shared().interface() else { exit(2) }
let name = CommandLine.arguments[1]
for _ in 1...4 {
    let found = ((try? wifi.scanForNetworks(withName: nil)) ?? []).filter { $0.ssid == name }
    if let network = found.max(by: { $0.rssiValue < $1.rssiValue }) {
        do {
            try wifi.associate(to: network, password: nil)
            exit(0)
        } catch {
            print("Joining \(name) failed: \(error.localizedDescription)")
        }
    }
    sleep(2)
}
exit(1)
