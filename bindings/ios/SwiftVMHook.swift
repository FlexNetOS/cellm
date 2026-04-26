//  SwiftVMHook.swift
//  Drop this ONE file into any Xcode project for SwiftVM hot-reload.
//  It polls /tmp/swiftvm-viewconfig.json and live-updates navigation titles,
//  labels, and text on screen — no other code changes required.
//  Only active in DEBUG Simulator builds. Complete no-op in Release.

#if DEBUG && targetEnvironment(simulator)
import UIKit
import Combine

@MainActor
final class SwiftVMHook {
    static let shared = SwiftVMHook()

    private var timer: Timer?
    private var config: [String: String] = [:]
    /// Maps on-screen title → config key (e.g. "Food Truck" → "Sidebar.navigationTitle")
    private var titleToKey: [String: String] = [:]
    private let path = "/tmp/swiftvm-viewconfig.json"
    private var lastData: Data?

    private init() {}

    func start() {
        guard timer == nil else { return }
        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.poll() }
        }
        RunLoop.main.add(timer!, forMode: .common)
        // Initial poll after a short delay to let UI settle
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.poll()
        }
    }

    private func poll() {
        guard let data = FileManager.default.contents(atPath: path) else { return }
        guard data != lastData else { return }
        lastData = data

        guard let newConfig = try? JSONDecoder().decode([String: String].self, from: data) else { return }

        let oldConfig = config
        config = newConfig

        // First time: build mapping from on-screen titles → config keys
        if titleToKey.isEmpty {
            buildInitialMapping()
        }

        // Detect changes: (1) value changes for existing keys, (2) key renames (Label/Text)
        for (key, newValue) in newConfig {
            let oldValue = oldConfig[key]
            if oldValue != newValue {
                applyChange(key: key, oldValue: oldValue, newValue: newValue)
            }
        }

        // Detect key renames: when a Label/Text key disappears and a new one appears
        // with the same struct+type prefix, it's a rename (the key embeds the string value).
        let removedKeys = Set(oldConfig.keys).subtracting(newConfig.keys)
        let addedKeys = Set(newConfig.keys).subtracting(oldConfig.keys)
        if !removedKeys.isEmpty && !addedKeys.isEmpty {
            for removed in removedKeys {
                guard let prefix = keyPrefix(removed) else { continue }
                let oldText = oldConfig[removed]!
                // Find an added key with the same prefix
                for added in addedKeys {
                    guard keyPrefix(added) == prefix else { continue }
                    let newText = newConfig[added]!
                    if oldText != newText {
                        applyChange(key: added, oldValue: oldText, newValue: newText)
                    }
                    break
                }
            }
        }
    }

    private func buildInitialMapping() {
        let navTitleKeys = config.filter { $0.key.hasSuffix(".navigationTitle") }
        let onScreenTitles = collectNavigationTitles()

        for title in onScreenTitles {
            if let match = navTitleKeys.first(where: { $0.value == title }) {
                titleToKey[title] = match.key
            }
        }
    }

    private func collectNavigationTitles() -> [String] {
        var titles: [String] = []
        for scene in UIApplication.shared.connectedScenes {
            guard let ws = scene as? UIWindowScene else { continue }
            for window in ws.windows {
                collectTitles(from: window.rootViewController, into: &titles)
                // Also collect titles from navigation bar UILabels (SwiftUI renders titles here)
                collectNavBarLabelTitles(in: window, into: &titles)
            }
        }
        return titles
    }

    private func collectTitles(from vc: UIViewController?, into titles: inout [String]) {
        guard let vc else { return }
        if let title = vc.navigationItem.title, !title.isEmpty {
            titles.append(title)
        }
        if let title = vc.title, !title.isEmpty, !titles.contains(title) {
            titles.append(title)
        }
        for child in vc.children {
            collectTitles(from: child, into: &titles)
        }
        if let presented = vc.presentedViewController {
            collectTitles(from: presented, into: &titles)
        }
    }

    /// Finds UILabels inside UINavigationBar that SwiftUI uses for large/inline titles
    private func collectNavBarLabelTitles(in view: UIView, into titles: inout [String]) {
        if let navBar = view as? UINavigationBar {
            collectLargeTitle(in: navBar, into: &titles)
            return
        }
        for sub in view.subviews {
            collectNavBarLabelTitles(in: sub, into: &titles)
        }
    }

    private func collectLargeTitle(in view: UIView, into titles: inout [String]) {
        if let label = view as? UILabel, let text = label.text, !text.isEmpty,
           !titles.contains(text), label.font.pointSize >= 28 {
            titles.append(text)
        }
        for sub in view.subviews {
            collectLargeTitle(in: sub, into: &titles)
        }
    }

    /// Extract the struct+type prefix from a key (e.g. "Sidebar.Label" from "Sidebar.Label.Orders")
    private func keyPrefix(_ key: String) -> String? {
        // Keys: "Struct.Label.value" or "Struct.Text.value" or "Struct.navigationTitle"
        if let labelRange = key.range(of: ".Label.") {
            return String(key[key.startIndex..<labelRange.upperBound])
        }
        if let textRange = key.range(of: ".Text.") {
            return String(key[key.startIndex..<textRange.upperBound])
        }
        return nil
    }

    private func applyChange(key: String, oldValue: String?, newValue: String) {
        if key.hasSuffix(".navigationTitle") {
            applyNavigationTitle(oldValue: oldValue, newValue: newValue, configKey: key)
        }
        // Labels and Text are matched by their full key (e.g. "Sidebar.Label.Truck")
        if key.contains(".Label.") || key.contains(".Text.") {
            applyLabelText(oldValue: oldValue, newValue: newValue)
        }
    }

    private func applyNavigationTitle(oldValue: String?, newValue: String, configKey: String) {
        for scene in UIApplication.shared.connectedScenes {
            guard let ws = scene as? UIWindowScene else { continue }
            for window in ws.windows {
                applyNavTitle(in: window.rootViewController, oldValue: oldValue, newValue: newValue, configKey: configKey)
                // SwiftUI renders .navigationTitle as UILabels in the nav bar — update those too
                applyNavBarLabels(in: window, oldValue: oldValue, newValue: newValue, configKey: configKey)
            }
        }
    }

    private func applyNavTitle(in vc: UIViewController?, oldValue: String?, newValue: String, configKey: String) {
        guard let vc else { return }

        let currentTitle = vc.navigationItem.title ?? vc.title

        // Match by: (1) exact old value, (2) known mapping, (3) reverse mapping
        let shouldUpdate: Bool
        if let old = oldValue, currentTitle == old {
            shouldUpdate = true
        } else if let ct = currentTitle, titleToKey[ct] == configKey {
            shouldUpdate = true
        } else {
            shouldUpdate = false
        }

        if shouldUpdate {
            // Remove old mapping, add new
            if let ct = currentTitle { titleToKey.removeValue(forKey: ct) }
            titleToKey[newValue] = configKey

            vc.navigationItem.title = newValue
            vc.title = newValue

            // Force navigation bar to re-layout
            vc.navigationController?.navigationBar.setNeedsLayout()
            vc.navigationController?.navigationBar.layoutIfNeeded()
        }

        for child in vc.children {
            applyNavTitle(in: child, oldValue: oldValue, newValue: newValue, configKey: configKey)
        }
        if let presented = vc.presentedViewController {
            applyNavTitle(in: presented, oldValue: oldValue, newValue: newValue, configKey: configKey)
        }
    }

    /// Walk the view hierarchy looking for UINavigationBar, then update labels inside it
    private func applyNavBarLabels(in view: UIView, oldValue: String?, newValue: String, configKey: String) {
        if let navBar = view as? UINavigationBar {
            updateNavBarTitleLabels(in: navBar, oldValue: oldValue, newValue: newValue, configKey: configKey)
            return
        }
        for sub in view.subviews {
            applyNavBarLabels(in: sub, oldValue: oldValue, newValue: newValue, configKey: configKey)
        }
    }

    /// Find and replace title UILabels inside a navigation bar (large title + inline title)
    private func updateNavBarTitleLabels(in view: UIView, oldValue: String?, newValue: String, configKey: String) {
        if let label = view as? UILabel {
            let text = label.text ?? ""
            let shouldReplace: Bool
            if let old = oldValue, text == old {
                shouldReplace = true
            } else if titleToKey[text] == configKey {
                shouldReplace = true
            } else {
                shouldReplace = false
            }
            if shouldReplace {
                if !text.isEmpty { titleToKey.removeValue(forKey: text) }
                titleToKey[newValue] = configKey
                label.text = newValue
            }
        }
        for sub in view.subviews {
            updateNavBarTitleLabels(in: sub, oldValue: oldValue, newValue: newValue, configKey: configKey)
        }
    }

    private func applyLabelText(oldValue: String?, newValue: String) {
        guard let old = oldValue, old != newValue else { return }
        for scene in UIApplication.shared.connectedScenes {
            guard let ws = scene as? UIWindowScene else { continue }
            for window in ws.windows {
                replaceLabels(in: window, oldText: old, newText: newValue)
            }
        }
    }

    private func replaceLabels(in view: UIView, oldText: String, newText: String) {
        if let label = view as? UILabel, label.text == oldText {
            label.text = newText
        }
        for sub in view.subviews {
            replaceLabels(in: sub, oldText: oldText, newText: newText)
        }
    }
}

// Called by SwiftVMHookBoot.m's __attribute__((constructor)) via dispatch_async.
// This guarantees the hook starts after the main run loop begins.
@_cdecl("_swiftvm_hook_init")
public func _swiftvm_hook_init() {
    Task { @MainActor in
        SwiftVMHook.shared.start()
    }
}
#endif
