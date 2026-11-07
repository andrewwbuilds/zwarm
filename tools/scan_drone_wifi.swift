// Prints the names of nearby S20 drone networks, strongest first.
import CoreWLAN
guard let wifi = CWWiFiClient.shared().interface() else { exit(1) }
let found = ((try? wifi.scanForNetworks(withName: nil)) ?? [])
    .filter { ($0.ssid ?? "").uppercased().hasPrefix("HITURBO_") }
    .sorted { $0.rssiValue > $1.rssiValue }
for network in found { print(network.ssid ?? "") }
