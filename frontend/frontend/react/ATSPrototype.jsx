import { useState, useRef } from "react";
import { ChevronRight, ChevronDown, X, Upload, Search, FileText } from "lucide-react";

const REQUIRED_SKILLS = [
  "Schedule and confirm appointments",
  "Draft and proofread correspondence",
  "Computer and technology knowledge",
  "MS Office",
  "Electronic mail",
  "Answer telephone and relay messages",
  "Maintain filing systems",
  "Order supplies",
  "Establish work priorities",
  "Customer service",
];

const FIRST_NAMES = ["Esther", "Michael", "Priya", "Carlos", "Aisha", "David", "Fatima", "James"];
const LAST_NAMES = ["Aziseh", "Okonkwo", "Rahman", "Delgado", "Bello", "Chen", "Hassan", "Turner"];
const EDU = ["Bachelor's", "Bachelor's", "Master's", "High School / Intermediate"];

function seededRandom(seed) {
  let x = Math.sin(seed) * 10000;
  return x - Math.floor(x);
}

function parseFileToCandidate(file, index) {
  const seed = file.name.length * 7 + index * 13 + Date.now() % 1000;
  const rnd = (n) => seededRandom(seed + n);
  const fname = FIRST_NAMES[Math.floor(rnd(1) * FIRST_NAMES.length)];
  const lname = LAST_NAMES[Math.floor(rnd(2) * LAST_NAMES.length)];
  const exp = +(rnd(3) * 8 + 0.5).toFixed(1);
  const edu = EDU[Math.floor(rnd(4) * EDU.length)];
  const expSource = rnd(9) > 0.5 ? "stated" : "computed";
  const roleCount = expSource === "computed" ? 1 + Math.floor(rnd(10) * 3) : null;

  const shuffled = [...REQUIRED_SKILLS].sort((a, b) => rnd(a.length) - rnd(b.length));
  const matchCount = 1 + Math.floor(rnd(5) * (REQUIRED_SKILLS.length - 1));
  const matched = shuffled.slice(0, matchCount);
  const missing = REQUIRED_SKILLS.filter((s) => !matched.includes(s));
  const score = Math.round((matched.length / REQUIRED_SKILLS.length) * 100 * 0.85);

  return {
    id: `cand_${seed}`,
    fileName: file.name,
    fullName: `${fname} ${lname}`,
    email: `${fname.toLowerCase()}${lname.toLowerCase()}${Math.floor(rnd(6) * 900)}@gmail.com`,
    phone: `+92 3${Math.floor(rnd(7) * 90 + 10)}-${Math.floor(rnd(8) * 9000000 + 1000000)}`,
    role: "administrative assistant",
    experience: exp,
    expSource,
    roleCount,
    education: edu,
    matched,
    missing,
    score,
    expanded: false,
  };
}

function ScoreBadge({ score }) {
  const tone =
    score >= 70 ? { bg: "#EAF4EC", fg: "#2B7A4B" } :
    score >= 40 ? { bg: "#FBF0DE", fg: "#9A6A16" } :
    { bg: "#FBEAE8", fg: "#B54B3F" };
  return (
    <span
      className="px-2.5 py-1 rounded-full text-xs font-semibold tabular-nums"
      style={{ background: tone.bg, color: tone.fg }}
    >
      {score}%
    </span>
  );
}

export default function ATSPrototype() {
  const [jobTitle, setJobTitle] = useState("administrative assistant");
  const [skillsText, setSkillsText] = useState(REQUIRED_SKILLS.join(", "));
  const [minExp, setMinExp] = useState(2);
  const [education, setEducation] = useState("High School / Intermediate");
  const [candidates, setCandidates] = useState([]);
  const [pendingFiles, setPendingFiles] = useState([]); // saved, not yet screened
  const [screening, setScreening] = useState(false);
  const [query, setQuery] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  // Step 1: save files only — nothing is parsed/scored yet.
  const handleFiles = (fileList) => {
    const files = Array.from(fileList).filter((f) =>
      /\.(pdf|docx)$/i.test(f.name)
    );
    const saved = files.map((f, i) => ({ id: `pend_${Date.now()}_${i}`, file: f }));
    setPendingFiles((prev) => [...prev, ...saved]);
  };

  const removePending = (id) => {
    setPendingFiles((prev) => prev.filter((p) => p.id !== id));
  };

  // Step 2: HR clicks "Screen CVs" — this is what actually runs extraction + scoring.
  const screenFiles = () => {
    if (pendingFiles.length === 0) return;
    setScreening(true);
    setTimeout(() => {
      const parsed = pendingFiles.map((p, i) =>
        parseFileToCandidate(p.file, candidates.length + i)
      );
      setCandidates((prev) => [...prev, ...parsed]);
      setPendingFiles([]);
      setScreening(false);
    }, 500);
  };

  const deleteAllCandidates = () => setCandidates([]);

  const toggleExpand = (id) => {
    setCandidates((prev) =>
      prev.map((c) => (c.id === id ? { ...c, expanded: !c.expanded } : c))
    );
  };

  const removeCandidate = (id) => {
    setCandidates((prev) => prev.filter((c) => c.id !== id));
  };

  const filtered = query
    ? candidates.filter(
        (c) =>
          c.fullName.toLowerCase().includes(query.toLowerCase()) ||
          c.matched.some((s) => s.toLowerCase().includes(query.toLowerCase()))
      )
    : candidates;

  const ranked = [...filtered].sort((a, b) => b.score - a.score);

  return (
    <div
      className="min-h-screen w-full"
      style={{ background: "#F6F4EF", fontFamily: "ui-sans-serif, system-ui" }}
    >
      <div className="max-w-7xl mx-auto p-6">
        <header className="mb-6 flex items-baseline justify-between">
          <div>
            <h1 className="text-xl font-semibold tracking-tight" style={{ color: "#1F2A24" }}>
              Candidate Screening
            </h1>
            <p className="text-sm" style={{ color: "#8A8578" }}>
              Set the role, drop in resumes, review matches.
            </p>
          </div>
          <span
            className="text-[11px] tracking-widest uppercase px-2 py-1 rounded"
            style={{ background: "#1F2A24", color: "#F6F4EF", fontFamily: "ui-monospace, monospace" }}
          >
            RAG · ChromaDB
          </span>
        </header>

        <div className="grid grid-cols-1 lg:grid-cols-[360px_1fr] gap-5">
          {/* LEFT: Job requirements */}
          <div
            className="rounded-xl p-5 border"
            style={{ background: "#FFFFFF", borderColor: "#E7E3D8" }}
          >
            <p
              className="text-[11px] tracking-widest uppercase mb-4"
              style={{ color: "#A39D8C", fontFamily: "ui-monospace, monospace" }}
            >
              Job Requirements
            </p>

            <label className="text-xs font-medium block mb-1" style={{ color: "#5B5646" }}>
              Role title
            </label>
            <input
              value={jobTitle}
              onChange={(e) => setJobTitle(e.target.value)}
              className="w-full mb-4 px-3 py-2 rounded-lg border text-sm outline-none"
              style={{ borderColor: "#E7E3D8" }}
            />

            <label className="text-xs font-medium block mb-1" style={{ color: "#5B5646" }}>
              Required skills
            </label>
            <textarea
              value={skillsText}
              onChange={(e) => setSkillsText(e.target.value)}
              rows={4}
              className="w-full mb-1 px-3 py-2 rounded-lg border text-sm outline-none resize-y"
              style={{ borderColor: "#E7E3D8" }}
            />
            <p className="text-[11px] mb-4" style={{ color: "#A39D8C" }}>
              Comma-separated. Matched per resume by the LLM.
            </p>

            <label className="text-xs font-medium block mb-1" style={{ color: "#5B5646" }}>
              Minimum years of experience
            </label>
            <input
              type="number"
              value={minExp}
              onChange={(e) => setMinExp(e.target.value)}
              className="w-full mb-4 px-3 py-2 rounded-lg border text-sm outline-none"
              style={{ borderColor: "#E7E3D8" }}
            />

            <label className="text-xs font-medium block mb-1" style={{ color: "#5B5646" }}>
              Required education
            </label>
            <select
              value={education}
              onChange={(e) => setEducation(e.target.value)}
              className="w-full mb-5 px-3 py-2 rounded-lg border text-sm outline-none bg-white"
              style={{ borderColor: "#E7E3D8" }}
            >
              <option>High School / Intermediate</option>
              <option>Bachelor's</option>
              <option>Master's</option>
            </select>

            <label className="text-xs font-medium block mb-1" style={{ color: "#5B5646" }}>
              Candidate resumes
            </label>
            <div
              onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleFiles(e.dataTransfer.files);
              }}
              onClick={() => fileInputRef.current?.click()}
              className="rounded-lg border-2 border-dashed flex flex-col items-center justify-center py-8 cursor-pointer transition-colors"
              style={{
                borderColor: dragOver ? "#1F2A24" : "#D9D4C6",
                background: dragOver ? "#F0EEE5" : "transparent",
              }}
            >
              <Upload size={18} style={{ color: "#A39D8C" }} />
              <p className="text-xs mt-2 text-center" style={{ color: "#8A8578" }}>
                Drop multiple PDFs/DOCX here, or click to browse
              </p>
              <input
                ref={fileInputRef}
                type="file"
                multiple
                accept=".pdf,.docx"
                className="hidden"
                onChange={(e) => handleFiles(e.target.files)}
              />
            </div>

            {pendingFiles.length > 0 && (
              <div className="mt-3 rounded-lg border p-3" style={{ borderColor: "#E7E3D8", background: "#FAF8F2" }}>
                <p className="text-[11px] tracking-widest uppercase mb-2" style={{ color: "#A39D8C" }}>
                  Saved — not yet screened ({pendingFiles.length})
                </p>
                <div className="space-y-1 mb-3 max-h-32 overflow-y-auto">
                  {pendingFiles.map((p) => (
                    <div key={p.id} className="flex items-center justify-between text-xs" style={{ color: "#5B5646" }}>
                      <span className="truncate" style={{ fontFamily: "ui-monospace, monospace" }}>{p.file.name}</span>
                      <button onClick={() => removePending(p.id)} style={{ color: "#B54B3F" }} aria-label={`Remove ${p.file.name}`}>
                        <X size={13} />
                      </button>
                    </div>
                  ))}
                </div>
                <button
                  onClick={screenFiles}
                  disabled={screening}
                  className="w-full py-2 rounded-lg text-sm font-medium disabled:opacity-60"
                  style={{ background: "#1F2A24", color: "#F6F4EF" }}
                >
                  {screening ? "Screening…" : `Screen ${pendingFiles.length} CV${pendingFiles.length > 1 ? "s" : ""}`}
                </button>
              </div>
            )}
          </div>

          {/* RIGHT: Search + Candidate table */}
          <div className="space-y-4">
            <div
              className="rounded-xl p-5 border"
              style={{ background: "#FFFFFF", borderColor: "#E7E3D8" }}
            >
              <p
                className="text-[11px] tracking-widest uppercase mb-3"
                style={{ color: "#A39D8C", fontFamily: "ui-monospace, monospace" }}
              >
                Ask about your candidates
              </p>
              <div className="flex gap-2">
                <div className="relative flex-1">
                  <Search size={15} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: "#A39D8C" }} />
                  <input
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder='e.g. "who has led a team" or "anyone with fintech experience"'
                    className="w-full pl-9 pr-3 py-2.5 rounded-lg border text-sm outline-none"
                    style={{ borderColor: "#E7E3D8" }}
                  />
                </div>
                <button
                  className="px-5 py-2.5 rounded-lg text-sm font-medium"
                  style={{ background: "#1F2A24", color: "#F6F4EF" }}
                >
                  Search
                </button>
              </div>
            </div>

            <div
              className="rounded-xl border overflow-hidden"
              style={{ background: "#FFFFFF", borderColor: "#E7E3D8" }}
            >
              <div className="px-5 pt-4 pb-2 flex items-center justify-between">
                <p
                  className="text-[11px] tracking-widest uppercase"
                  style={{ color: "#A39D8C", fontFamily: "ui-monospace, monospace" }}
                >
                  Candidates ({ranked.length})
                </p>
                {candidates.length > 0 && (
                  <div className="flex items-center gap-3">
                    <p className="text-[11px]" style={{ color: "#A39D8C" }}>
                      ranked best → worst
                    </p>
                    <button
                      onClick={deleteAllCandidates}
                      className="text-[11px] tracking-wide uppercase px-2 py-1 rounded hover:bg-[#FBEAE8]"
                      style={{ color: "#B54B3F", fontFamily: "ui-monospace, monospace" }}
                    >
                      Delete all
                    </button>
                  </div>
                )}
              </div>

              {ranked.length === 0 ? (
                <div className="py-16 flex flex-col items-center gap-2">
                  <FileText size={22} style={{ color: "#D9D4C6" }} />
                  <p className="text-sm" style={{ color: "#A39D8C" }}>
                    No resumes yet — upload some on the left.
                  </p>
                </div>
              ) : (
                <div>
                  <div
                    className="grid px-5 py-2 text-[11px] tracking-wide uppercase"
                    style={{
                      gridTemplateColumns: "28px 2fr 1.4fr 0.8fr 1fr 0.7fr 32px",
                      color: "#A39D8C",
                      borderTop: "1px solid #EFEBDF",
                      borderBottom: "1px solid #EFEBDF",
                    }}
                  >
                    <span />
                    <span>Candidate file</span>
                    <span>Role</span>
                    <span>Experience</span>
                    <span>Education</span>
                    <span>Score</span>
                    <span />
                  </div>

                  {ranked.map((c) => (
                    <div key={c.id}>
                      <div
                        className="grid items-center px-5 py-3 text-sm cursor-pointer hover:bg-[#FAF8F2]"
                        style={{
                          gridTemplateColumns: "28px 2fr 1.4fr 0.8fr 1fr 0.7fr 32px",
                          borderBottom: "1px solid #F1EEE3",
                        }}
                        onClick={() => toggleExpand(c.id)}
                      >
                        <span style={{ color: "#A39D8C" }}>
                          {c.expanded ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                        </span>
                        <span className="truncate pr-2" style={{ fontFamily: "ui-monospace, monospace", color: "#1F2A24" }}>
                          {c.fileName}
                        </span>
                        <span style={{ color: "#5B5646" }}>{c.role}</span>
                        <span style={{ color: "#5B5646" }}>{c.experience} yrs</span>
                        <span style={{ color: "#5B5646" }}>{c.education}</span>
                        <span><ScoreBadge score={c.score} /></span>
                        <button
                          onClick={(e) => { e.stopPropagation(); removeCandidate(c.id); }}
                          className="rounded p-1 hover:bg-[#FBEAE8]"
                          style={{ color: "#B54B3F" }}
                          aria-label={`Remove ${c.fullName}`}
                        >
                          <X size={15} />
                        </button>
                      </div>

                      {c.expanded && (
                        <div
                          className="px-5 py-5 grid grid-cols-1 md:grid-cols-2 gap-6"
                          style={{ background: "#FAF8F2", borderBottom: "1px solid #F1EEE3" }}
                        >
                          <div>
                            <p className="text-[11px] tracking-widest uppercase mb-2" style={{ color: "#A39D8C" }}>
                              Skills — {c.matched.length} of {REQUIRED_SKILLS.length} matched
                            </p>
                            <div className="flex flex-wrap gap-1.5 mb-4">
                              {c.matched.map((s) => (
                                <span
                                  key={s}
                                  className="text-xs px-2.5 py-1 rounded-full"
                                  style={{ background: "#EAF4EC", color: "#2B7A4B" }}
                                >
                                  {s}
                                </span>
                              ))}
                            </div>
                            {c.missing.length > 0 && (
                              <>
                                <p className="text-[11px] tracking-widest uppercase mb-2" style={{ color: "#A39D8C" }}>
                                  Missing skills
                                </p>
                                <div className="flex flex-wrap gap-1.5">
                                  {c.missing.map((s) => (
                                    <span
                                      key={s}
                                      className="text-xs px-2.5 py-1 rounded-full"
                                      style={{ background: "#FBEAE8", color: "#B54B3F" }}
                                    >
                                      {s}
                                    </span>
                                  ))}
                                </div>
                              </>
                            )}
                          </div>

                          <div>
                            <p className="text-[11px] tracking-widest uppercase mb-2" style={{ color: "#A39D8C" }}>
                              Requirement checklist
                            </p>
                            <p className="text-xs mb-3" style={{ color: "#A39D8C" }}>
                              Score is based on skill match only; shown here for reference.
                            </p>
                            <div className="text-sm mb-2" style={{ color: "#1F2A24" }}>
                              <strong>Education</strong> — requires {education}
                              <div className="text-xs" style={{ color: "#8A8578" }}>{c.education}</div>
                            </div>
                            <div className="text-sm mb-4" style={{ color: "#1F2A24" }}>
                              <strong>Experience</strong> — requires {minExp}+ yrs
                              <div className="text-xs" style={{ color: "#8A8578" }}>
                                {c.experience} yrs found —{" "}
                                {c.expSource === "stated"
                                  ? "stated directly in resume"
                                  : `computed from ${c.roleCount} role${c.roleCount > 1 ? "s" : ""} (dates normalized)`}
                              </div>
                            </div>
                            <p className="text-[11px] tracking-widest uppercase mb-2" style={{ color: "#A39D8C" }}>
                              Contact
                            </p>
                            <div className="text-sm" style={{ color: "#1F2A24" }}>{c.email}</div>
                            <div className="text-sm" style={{ color: "#1F2A24" }}>{c.phone}</div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
