import SwiftUI
import UniformTypeIdentifiers

struct ConcurrentChatView: View {
    struct Msg: Identifiable {
        let id = UUID()
        let role: String
        var text: String
    }
    struct Thread: Identifiable {
        let id = UUID()
        var title: String
        var msgs: [Msg] = []
        var sessionId: UInt64 = 0
        var generating = false
        var tokCount = 0
        var err: String?
    }

    @State private var threads: [Thread] = []
    @State private var selId: UUID?
    @State private var engine: CellmConcurrentEngine?
    @State private var tokenizer: CellmTokenizer?
    @State private var decodeTask: Task<Void, Never>?
    @State private var showSettings = false
    @State private var showMgr = false
    @State private var temp: Double = 0.2
    @State private var maxTok: Int = 200
    @State private var backend: CellmBackend = .metal
    @State private var initing = false
    @State private var err: String?
    @State private var backendLabel = ""
    @State private var modelURL: URL?
    @State private var tokURL: URL?
    @State private var modelLabel = ""
    @State private var newTitle = ""
    @State private var showNew = false
    @State private var pick: PickerTarget?
    @Environment(\.colorScheme) private var scheme

    private enum PickerTarget: String, Identifiable {
        case llmModel, tokenizer
        var id: String { rawValue }
        var allowedTypes: [UTType] {
            self == .tokenizer ? [.json] : [.item]
        }
    }

    private var selIdx: Int? { threads.firstIndex(where: { $0.id == selId }) }

    var body: some View {
        ZStack {
            Color(.systemBackground).ignoresSafeArea()
            VStack(spacing: 0) {
                header
                if let e = err, !e.isEmpty {
                    Text(e).font(.footnote).foregroundStyle(.red)
                        .padding(.horizontal, 16).padding(.vertical, 6)
                        .background(Color.red.opacity(0.08))
                        .clipShape(RoundedRectangle(cornerRadius: 10))
                        .padding(.horizontal, 16)
                }
                if threads.isEmpty || engine == nil {
                    Spacer(); emptyState; Spacer()
                } else {
                    threadBody
                }
                tabBar
            }
        }
        .sheet(item: $pick) { target in
            DocumentPicker(allowed: target.allowedTypes) { url in
                if target == .llmModel { modelURL = persist(url, "picked/concurrent/llm") }
                else { tokURL = persist(url, "picked/concurrent/tokenizer") }
                invalidate(); initEngine()
            }
        }
        .sheet(isPresented: $showSettings) {
            GenerationSettingsSheet(temperature: $temp, maxNewTokens: $maxTok, selectedBackend: $backend)
        }
        .sheet(isPresented: $showMgr) {
            SessionMgrSheet(threads: threads, kv: engine?.kvStats() ?? (0,0), backend: backendLabel)
        }
        .sheet(isPresented: $showNew) {
            NavigationStack {
                Form { TextField("Thread Name", text: $newTitle) }
                .navigationTitle("New Thread")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .cancellationAction) { Button("Cancel") { showNew = false } }
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Create") {
                            createThread(newTitle.isEmpty ? "Thread \(threads.count+1)" : newTitle)
                            showNew = false; newTitle = ""
                        }
                    }
                }
            }
            .presentationDetents([.medium])
        }
        .onAppear { restore(); initEngine() }
        .onDisappear { invalidate() }
        .onChange(of: modelURL) { _ in initEngine() }
        .onChange(of: tokURL) { _ in initEngine() }
        .onChange(of: backend) { _ in invalidate(); initEngine() }
    }

    // MARK: Header
    private var header: some View {
        VStack(spacing: 10) {
            HStack {
                Menu {
                    Section("Presets") {
                        Button("Gemma 4") { loadPreset(DemoAssetLinks.gemma4E2BFileName, DemoAssetLinks.gemma4E2BTokenizerFileName, "Gemma-4") }
                        Button("Qwen 2.5") { loadPreset(DemoAssetLinks.qwen25FileName, DemoAssetLinks.qwen25TokenizerFileName, "Qwen2.5") }
                        Button("NanoWhale (MLA+MoE)") { loadPreset(DemoAssetLinks.nanowhaleFileName, DemoAssetLinks.nanowhaleTokenizerFileName, "NanoWhale") }
                        Button("LFM 2.5 (Liquid)") { loadPreset(DemoAssetLinks.lfm25FileName, DemoAssetLinks.lfm25TokenizerFileName, "LFM-2.5") }
                        Button("SmolLM2") { loadPreset(DemoAssetLinks.smollm2FileName, DemoAssetLinks.smollm2TokenizerFileName, "SmolLM2") }
                    }
                    Section("Advanced") {
                        Button("Pick LLM...") { pick = .llmModel }
                        Button("Pick Tokenizer...") { pick = .tokenizer }
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: engine == nil ? "arrow.down.circle" : "checkmark.circle.fill")
                            .foregroundStyle(engine == nil ? Color.secondary : .green)
                        Text(modelLabel.isEmpty ? "Select Model" : modelLabel)
                            .font(.subheadline.bold())
                        Image(systemName: "chevron.down").font(.caption2)
                    }
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(Color(.systemGray6)).clipShape(Capsule())
                }
                Spacer()
                HStack(spacing: 12) {
                    Button { showMgr = true } label: { Image(systemName: "gauge.with.dots.needle.67percent").font(.title3) }
                    Button { showSettings = true } label: { Image(systemName: "slider.horizontal.3").font(.title3) }
                }
                .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 16).padding(.top, 8)

            if initing {
                HStack(spacing: 8) {
                    ProgressView().scaleEffect(0.8)
                    Text("Initializing...").font(.footnote.bold()).foregroundStyle(.cyan)
                }
                .padding(.horizontal, 14).padding(.vertical, 6)
                .background(Color.cyan.opacity(0.12)).clipShape(Capsule())
            }
            if !backendLabel.isEmpty {
                HStack(spacing: 4) {
                    Image(systemName: backendLabel == "metal" ? "bolt.fill" : "cpu").font(.caption2)
                    Text(backendLabel.uppercased()).font(.caption2.bold())
                }
                .foregroundStyle(backendLabel == "metal" ? .green : Color.secondary)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background((backendLabel == "metal" ? Color.green : Color.gray).opacity(0.12))
                .clipShape(Capsule())
            }
        }
        .padding(.bottom, 8)
    }

    // MARK: Thread Body
    @ViewBuilder
    private var threadBody: some View {
        if let idx = selIdx {
            VStack(spacing: 0) {
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 10) {
                            ForEach(threads[idx].msgs) { msg in bubble(msg) }
                        }
                        .padding(12)
                    }
                    .onChange(of: threads[idx].msgs.count) { _ in
                        if let last = threads[idx].msgs.last {
                            withAnimation(.easeOut(duration: 0.15)) {
                                proxy.scrollTo(last.id, anchor: .bottom)
                            }
                        }
                    }
                }
                if threads[idx].generating {
                    HStack {
                        ProgressView().scaleEffect(0.7)
                        Text("Generating... \(threads[idx].tokCount) tokens").font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button("Stop") { stop(threads[idx].id) }
                            .font(.caption.bold()).foregroundStyle(.red)
                    }
                    .padding(.horizontal, 16).padding(.vertical, 6)
                }
                if let e = threads[idx].err, !e.isEmpty {
                    Text(e).font(.footnote).foregroundStyle(.red)
                        .padding(.horizontal, 12).padding(.vertical, 6)
                        .background(Color.red.opacity(0.08))
                        .clipShape(RoundedRectangle(cornerRadius: 8))
                        .padding(.horizontal, 16)
                }
                ThreadComposer(isGenerating: threads[idx].generating) { text in
                    send(idx: threads[idx].id, text: text)
                }
            }
        } else {
            Spacer()
            Text("Select a thread").foregroundStyle(.secondary)
            Spacer()
        }
    }

    private func bubble(_ msg: Msg) -> some View {
        VStack(alignment: msg.role == "User" ? .trailing : .leading, spacing: 4) {
            Text(msg.role).font(.caption2).foregroundStyle(.secondary)
            Text(msg.text)
                .padding(.horizontal, 14).padding(.vertical, 10)
                .background(msg.role == "User" ? Color.accentColor.opacity(scheme == .dark ? 0.35 : 0.18) : Color(.secondarySystemGroupedBackground))
                .foregroundStyle(.primary)
                .clipShape(RoundedRectangle(cornerRadius: 18))
                .textSelection(.enabled)
        }
        .frame(maxWidth: .infinity, alignment: msg.role == "User" ? .trailing : .leading)
    }

    // MARK: Tab Bar
    private var tabBar: some View {
        HStack(spacing: 0) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    ForEach(threads) { t in
                        Button { selId = t.id } label: {
                            HStack(spacing: 4) {
                                if t.generating {
                                    Image(systemName: "circle.fill").font(.system(size: 6)).foregroundStyle(.green)
                                }
                                Text(t.title).font(.caption.bold()).lineLimit(1)
                            }
                            .padding(.horizontal, 12).padding(.vertical, 8)
                            .background(selId == t.id ? Color.accentColor.opacity(0.15) : Color(.systemGray6))
                            .foregroundStyle(selId == t.id ? .primary : .secondary)
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                            .overlay(RoundedRectangle(cornerRadius: 10).stroke(selId == t.id ? Color.accentColor.opacity(0.4) : Color.clear, lineWidth: 1))
                        }
                        .contextMenu {
                            Button(role: .destructive) { delete(t.id) } label: { Label("Delete", systemImage: "trash") }
                        }
                    }
                }
                .padding(.horizontal, 12)
            }
            Button { showNew = true } label: {
                Image(systemName: "plus.circle.fill").font(.title3).foregroundColor(.accentColor)
                    .padding(.horizontal, 12)
            }
        }
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
        .overlay(Rectangle().frame(height: 0.5).foregroundStyle(Color(.separator)), alignment: .top)
    }

    private var emptyState: some View {
        VStack(spacing: 16) {
            Image(systemName: "bubble.left.and.bubble.right.fill").font(.system(size: 48)).foregroundStyle(.secondary)
            Text("Concurrent Chat").font(.title2.bold())
            Text("Load a model and create multiple chat threads sharing one engine. Tokens decode in round-robin order.")
                .font(.subheadline).multilineTextAlignment(.center).foregroundStyle(.secondary)
                .padding(.horizontal, 32)
            Button("Create First Thread") { showNew = true }
                .buttonStyle(.borderedProminent)
                .disabled(initing)
        }
    }

    // MARK: Logic
    private func createThread(_ title: String) {
        guard let eng = engine else { return }
        let t = Thread(title: title, msgs: [Msg(role: "Assistant", text: "Hi! Ready to chat.")], sessionId: eng.createSession())
        threads.append(t)
        if selId == nil { selId = t.id }
    }

    private func delete(_ id: UUID) {
        guard let i = threads.firstIndex(where: { $0.id == id }) else { return }
        if threads[i].sessionId != 0 { engine?.cancelSession(threads[i].sessionId) }
        threads.remove(at: i)
        if selId == id { selId = threads.first?.id }
    }

    private func stop(_ id: UUID) {
        guard let i = threads.firstIndex(where: { $0.id == id }) else { return }
        threads[i].generating = false
        if threads[i].sessionId != 0 { engine?.cancelSession(threads[i].sessionId) }
    }

    private func send(idx: UUID, text: String) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }
        guard let i = threads.firstIndex(where: { $0.id == idx }) else { return }
        guard let eng = engine, let tok = tokenizer else { return }

        threads[i].msgs.append(Msg(role: "User", text: clean))
        threads[i].msgs.append(Msg(role: "Assistant", text: ""))
        threads[i].generating = true
        threads[i].tokCount = 0
        threads[i].err = nil

        let prompt = buildPrompt(threads[i].msgs)
        let proc = ModelDataProcessorFactory.make(tokenizerURL: tok.tokenizerURL, modelURL: modelURL)
        let wrapped = proc.wrapPrompt(prompt, system: nil)
        let toks: [UInt32]
        do { toks = try tok.encode(wrapped) }
        catch {
            threads[i].err = "Tokenize: \(error)"
            threads[i].generating = false
            return
        }

        let sid = threads[i].sessionId
        let maxT = maxTok
        let tempV = temp

        Task.detached(priority: .userInitiated) {
            do {
                let (first, _) = try eng.submitTokensCached(session: sid, tokens: toks)
                let piece = try tok.decodeOne(first)
                let stop = proc.isStopPiece(piece)

                await MainActor.run {
                    guard let idx2 = self.threads.firstIndex(where: { $0.id == idx }) else { return }
                    if let li = self.threads[idx2].msgs.indices.last,
                       self.threads[idx2].msgs[li].role == "Assistant" {
                        self.threads[idx2].msgs[li].text += piece
                    }
                    self.threads[idx2].tokCount = 1
                    if stop || self.threads[idx2].tokCount >= maxT {
                        self.threads[idx2].generating = false
                    }
                }
                self.ensureLoop()
            } catch {
                await MainActor.run {
                    guard let idx2 = self.threads.firstIndex(where: { $0.id == idx }) else { return }
                    self.threads[idx2].err = String(describing: error)
                    self.threads[idx2].generating = false
                }
            }
        }
    }

    private func ensureLoop() {
        guard decodeTask == nil || decodeTask!.isCancelled else { return }
        
        // Capture a strong reference to the engine if it exists.
        // If it doesn't exist, we can't start a loop anyway.
        guard let eng = self.engine else { return }
        let tok = self.tokenizer

        decodeTask = Task.detached(priority: .userInitiated) {
            while !Task.isCancelled {
                let active = await MainActor.run { self.threads.contains(where: { $0.generating }) }
                guard active else { try? await Task.sleep(nanoseconds: 20_000_000); continue }

                do {
                    guard let res = try eng.stepDecode() else {
                        try? await Task.sleep(nanoseconds: 5_000_000); continue
                    }
                    guard let piece = try? tok?.decodeOne(res.token) else { continue }

                    let modelName = await MainActor.run { self.modelURL?.lastPathComponent.lowercased() ?? "" }
                    let tokName = await MainActor.run { self.tokURL?.lastPathComponent.lowercased() ?? "" }
                    let stop = Self.isStopPiece(piece, modelName: modelName, tokenizerName: tokName)

                    await MainActor.run {
                        guard let i = self.threads.firstIndex(where: { $0.sessionId == res.session && $0.generating }) else { return }
                        if let li = self.threads[i].msgs.indices.last,
                           self.threads[i].msgs[li].role == "Assistant" {
                            self.threads[i].msgs[li].text += piece
                        }
                        self.threads[i].tokCount += 1
                        let currentTokCount = self.threads[i].tokCount
                        let limit = self.maxTok
                        if stop || currentTokCount >= limit {
                            self.threads[i].generating = false
                        }
                    }
                } catch {
                    await MainActor.run { self.err = String(describing: error) }
                }
            }
        }
    }

    private static func isStopPiece(_ piece: String, modelName: String, tokenizerName: String) -> Bool {
        let p = piece.trimmingCharacters(in: .whitespacesAndNewlines)
        if tokenizerName.contains("gemma-4") || modelName.contains("gemma-4") {
            return p == "<turn|>" || p == "<|endoftext|>"
        }
        if tokenizerName.contains("gemma") || modelName.contains("gemma") {
            return p == "<end_of_turn>" || p == "<|endoftext|>"
        }
        if tokenizerName.contains("qwen") || modelName.contains("qwen") || tokenizerName.contains("lfm") || modelName.contains("lfm") {
            return p == "<|endoftext|>" || p.hasSuffix("<|endoftext|>") || p == "eos" || p.hasSuffix("eos")
        }
        if tokenizerName.contains("smollm") || modelName.contains("smollm") {
            return p == "<end_of_utterance>" || p == "<|endoftext|>"
        }
        return p == "<|endoftext|>"
    }

    private func buildPrompt(_ msgs: [Msg]) -> String {
        var lines: [String] = ["You are a helpful assistant."]
        for m in msgs {
            if m.role == "User" { lines.append("User: \(m.text)") }
            else if !m.text.isEmpty { lines.append("Assistant: \(m.text)") }
        }
        return lines.joined(separator: "\n")
    }

    // MARK: Engine lifecycle
    private func initEngine() {
        guard let m = modelURL, let t = tokURL else { return }
        invalidate()
        initing = true
        Task {
            do {
                let tok = try CellmTokenizer(tokenizerURL: t)
                let topK: UInt32 = temp < 0.05 ? 1 : 40
                let eng = try CellmConcurrentEngine(modelURL: m, tokenizer: tok, topK: topK, temperature: Float(temp), backend: backend)
                await MainActor.run {
                    self.tokenizer = tok
                    self.engine = eng
                    self.backendLabel = eng.activeBackend
                    self.initing = false
                    self.err = nil
                }
            } catch {
                await MainActor.run { self.initing = false; self.err = String(describing: error) }
            }
        }
    }

    private func invalidate() {
        for i in threads.indices {
            if threads[i].sessionId != 0 { engine?.cancelSession(threads[i].sessionId) }
            threads[i].sessionId = 0
            threads[i].generating = false
        }
        decodeTask?.cancel(); decodeTask = nil
        engine = nil; tokenizer = nil; backendLabel = ""
    }

    // MARK: Presets / persistence
    private func restore() {
        if let mp = UserDefaults.standard.string(forKey: "cellm.concurrent.model"),
           let tp = UserDefaults.standard.string(forKey: "cellm.concurrent.tokenizer") {
            let m = URL(fileURLWithPath: mp), t = URL(fileURLWithPath: tp)
            if FileManager.default.fileExists(atPath: m.path) && FileManager.default.fileExists(atPath: t.path) {
                modelURL = m; tokURL = t; modelLabel = "Restored"
            }
        }
        if modelURL == nil || tokURL == nil {
            tryLoadPreset(DemoAssetLinks.gemma4E2BFileName, DemoAssetLinks.gemma4E2BTokenizerFileName, "Gemma-4")
        }
        if modelURL == nil || tokURL == nil {
            tryLoadPreset(DemoAssetLinks.qwen25FileName, DemoAssetLinks.qwen25TokenizerFileName, "Qwen2.5")
        }
        if modelURL == nil || tokURL == nil {
            tryLoadPreset(DemoAssetLinks.smollm2FileName, DemoAssetLinks.smollm2TokenizerFileName, "SmolLM2")
        }
    }

    private func tryLoadPreset(_ modelFile: String, _ tokFile: String, _ label: String) {
        let m = RemoteAssets.existingDocumentsFile(fileName: modelFile)
        let t = RemoteAssets.existingDocumentsFile(fileName: tokFile)
        if let m, let t { modelURL = m; tokURL = t; modelLabel = label }
    }

    private func loadPreset(_ modelFile: String, _ tokFile: String, _ label: String) {
        tryLoadPreset(modelFile, tokFile, label)
        initEngine()
    }

    private func persist(_ url: URL, _ subdir: String) -> URL? {
        do {
            let name = url.lastPathComponent
            let dest = RemoteAssets.documentsURL(fileName: "\(subdir)/\(name)")
            let dir = dest.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            if FileManager.default.fileExists(atPath: dest.path) { try FileManager.default.removeItem(at: dest) }
            try FileManager.default.copyItem(at: url, to: dest)
            return dest
        } catch { return nil }
    }
}

// MARK: - Thread Composer

private struct ThreadComposer: View {
    let isGenerating: Bool
    let onSend: (String) -> Void
    @State private var text = ""
    @FocusState private var focused: Bool

    var body: some View {
        HStack(alignment: .bottom, spacing: 12) {
            ZStack(alignment: .leading) {
                if text.isEmpty {
                    Text("Type prompt...").foregroundStyle(.secondary)
                        .padding(.horizontal, 12).padding(.vertical, 14)
                }
                TextEditor(text: $text)
                    .scrollContentBackground(.hidden)
                    .background(Color.clear)
                    .foregroundStyle(.primary)
                    .padding(.horizontal, 8).padding(.vertical, 8)
                    .frame(minHeight: 44, maxHeight: 96)
                    .focused($focused)
            }
            .background(Color(.secondarySystemGroupedBackground))
            .clipShape(RoundedRectangle(cornerRadius: 24))
            .overlay(RoundedRectangle(cornerRadius: 24).stroke(Color(.separator).opacity(0.35), lineWidth: 1))

            Button {
                let t = text; text = ""
                onSend(t)
            } label: {
                Image(systemName: "paperplane.fill").font(.title3)
                    .padding(12)
                    .background(text.isEmpty || isGenerating ? Color(.secondarySystemGroupedBackground) : Color.accentColor)
                    .foregroundColor(text.isEmpty || isGenerating ? .secondary : .white)
                    .clipShape(Circle())
            }
            .disabled(text.isEmpty || isGenerating)
        }
        .padding(.horizontal, 16).padding(.bottom, 24)
    }
}

// MARK: - Session Manager Sheet

private struct SessionMgrSheet: View {
    let threads: [ConcurrentChatView.Thread]
    let kv: (used: UInt32, free: UInt32)
    let backend: String
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section("Engine") {
                    HStack {
                        Text("Backend")
                        Spacer()
                        Text(backend.uppercased()).font(.caption.bold()).foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("KV Used / Free")
                        Spacer()
                        Text("\(kv.used) / \(kv.free)").font(.caption.bold()).foregroundStyle(.secondary)
                    }
                    if kv.free > 0 {
                        ProgressView(value: Double(kv.used), total: Double(kv.used + kv.free))
                            .tint(.blue)
                    }
                }

                Section("Active Sessions (\(threads.count))") {
                    ForEach(threads) { t in
                        VStack(alignment: .leading, spacing: 6) {
                            HStack {
                                Text(t.title).font(.headline)
                                Spacer()
                                if t.generating {
                                    Label("Active", systemImage: "bolt.fill")
                                        .font(.caption)
                                        .foregroundStyle(.green)
                                }
                            }
                            Text("Session: \(t.sessionId)")
                                .font(.caption2).foregroundStyle(.secondary).monospaced()
                            Text("Messages: \(t.msgs.count)  Tokens: \(t.tokCount)")
                                .font(.caption).foregroundStyle(.secondary)
                            if let e = t.err, !e.isEmpty {
                                Text(e).font(.caption2).foregroundStyle(.red)
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
            .navigationTitle("Session Manager")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
            }
        }
        .presentationDetents([.medium, .large])
    }
}

// MARK: - Helper for thread list access

extension ConcurrentChatView.Thread: @unchecked Sendable {}
extension ConcurrentChatView.Msg: @unchecked Sendable {}
