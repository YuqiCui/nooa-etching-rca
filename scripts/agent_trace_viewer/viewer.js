"use strict";
(() => {
  // 所有轨迹内容均作为文本展示，不执行消息中的 HTML、脚本或工具调用。
  const trace = JSON.parse(document.getElementById("trace-data").textContent);
  const $ = (id) => document.getElementById(id);
  const actors = new Map(trace.actors.map((actor, index) => [actor.id, { ...actor, index }]));
  const events = new Map(trace.events.map((event) => [event.id, event]));
  const cache = new Map();
  const positions = new Map();
  const cards = new Map();
  const colors = ["#177b72", "#3476a5", "#967049", "#8561a5", "#4c8775", "#b06d7f"];
  const clockWidth = 98;
  const formatter = new Intl.DateTimeFormat("zh-CN", {
    timeZone: trace.timezone, hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
  const dateFormatter = new Intl.DateTimeFormat("zh-CN", {
    timeZone: trace.timezone, year: "numeric", month: "2-digit", day: "2-digit",
  });
  let laneWidth = 300;
  let selected = null;
  let activeTab = null;
  let anchors = [];
  let returnFocus = null;

  function node(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  }

  function blob(id) {
    if (id === undefined || id === null) return null;
    if (!cache.has(id)) cache.set(id, JSON.parse(trace.blobs[id]));
    return cache.get(id);
  }

  function pretty(value) {
    return typeof value === "string" ? value : JSON.stringify(value, null, 2);
  }

  function time(value) {
    return formatter.format(new Date(value));
  }

  function duration(milliseconds) {
    if (milliseconds < 1000) return `${Math.round(milliseconds)} ms`;
    const seconds = milliseconds / 1000;
    return seconds < 60 ? `${seconds.toFixed(1)} 秒` : `${Math.floor(seconds / 60)}分${Math.round(seconds % 60)}秒`;
  }

  function statusText(event) {
    return ({ error: "异常", warning: "输出含异常", pending: "未结束" })[event.status] ||
      (event.end === null ? "" : duration(Math.max(0, event.end - event.start)));
  }

  function messages(event) {
    return (event.messages || []).map((id) => blob(id));
  }

  function showPrompt(actorId) {
    const requests = trace.events.filter((event) => event.actor === actorId && event.kind === "model");
    const current = selected && events.get(selected);
    const event = current?.actor === actorId && current.kind === "model" ? current : requests[0];
    if (event) openDetail(event, "system");
  }

  function renderHeadings() {
    const heading = $("lane-headings");
    heading.replaceChildren();
    const clock = node("div", "clock-heading", "时间 ↓");
    clock.title = "所有列共享时间顺序；密集事件展开显示，间距不代表等比例时长。";
    heading.append(clock);
    for (const actor of actors.values()) {
      const lane = node("div", "lane-heading");
      lane.style.setProperty("--actor-color", colors[actor.index % colors.length]);
      lane.title = actor.id + (actor.parent ? `\n上级：${actor.parent}` : "");
      lane.append(node("span", "actor-dot"), node("span", "lane-name", actor.name));
      const prompt = node("button", "prompt-button", "Prompt");
      prompt.type = "button";
      prompt.disabled = !trace.events.some((event) => event.actor === actor.id && event.kind === "model");
      prompt.setAttribute("aria-label", `查看 ${actor.name} 的 System 和 User prompt`);
      prompt.addEventListener("click", () => showPrompt(actor.id));
      lane.append(prompt);
      heading.append(lane);
    }
  }

  function positionAt(at) {
    if (at <= anchors[0].at) return anchors[0].y;
    let low = 0;
    let high = anchors.length - 1;
    while (low + 1 < high) {
      const middle = Math.floor((low + high) / 2);
      if (anchors[middle].at <= at) low = middle;
      else high = middle;
    }
    const left = anchors[low];
    const right = anchors[high];
    return left.y + (right.y - left.y) * Math.min(1, (at - left.at) / (right.at - left.at || 1));
  }

  function renderTimeline() {
    const viewport = $("viewport");
    const stage = $("stage");
    const spacing = Number($("spacing").value);
    const cardHeight = 46 + 16 * (spacing - 0.8);
    const step = cardHeight + 12;
    laneWidth = Math.max(275, (viewport.clientWidth - clockWidth) / actors.size);
    document.documentElement.style.setProperty("--lane-width", `${laneWidth}px`);
    const width = clockWidth + laneWidth * actors.size;
    $("lane-headings").style.width = `${width}px`;
    stage.style.width = `${width}px`;
    stage.replaceChildren();
    cards.clear();
    positions.clear();
    const groups = new Map();
    for (const event of trace.events) {
      if (!groups.has(event.start)) groups.set(event.start, []);
      groups.get(event.start).push(event);
    }
    groups.set(trace.start, groups.get(trace.start) || []);
    groups.set(trace.end, groups.get(trace.end) || []);
    anchors = [];
    let y = 24;
    let previous = trace.start;
    for (const [at, group] of [...groups].sort((a, b) => a[0] - b[0])) {
      y += Math.min(150, Math.max(0, at - previous) * 0.045 * spacing);
      anchors.push({ at, y });
      const used = new Map();
      for (const event of group) {
        const slot = used.get(event.actor) || 0;
        positions.set(event.id, { y: y + slot * step, x: clockWidth + actors.get(event.actor).index * laneWidth });
        used.set(event.actor, slot + 1);
      }
      y += Math.max(1, ...used.values()) * step;
      previous = at;
    }
    stage.style.height = `${y + 50}px`;
    for (const actor of actors.values()) {
      const background = node("div", `lane-background${actor.parent ? "" : " router"}`);
      background.style.left = `${clockWidth + actor.index * laneWidth}px`;
      background.style.width = `${laneWidth}px`;
      background.dataset.actor = actor.id;
      stage.append(background);
    }
    stage.append(node("div", "clock-strip"));
    for (const anchor of anchors) {
      const tick = node("time", "time-tick", time(anchor.at));
      tick.dateTime = new Date(anchor.at).toISOString();
      tick.title = `${new Date(anchor.at).toISOString()} · +${duration(anchor.at - trace.start)}`;
      tick.style.top = `${anchor.y + 3}px`;
      stage.append(tick);
    }
    for (const event of trace.events) {
      const position = positions.get(event.id);
      if (["model", "tool"].includes(event.kind) && event.end !== null) {
        const span = node("div", `activity-span ${event.kind} ${event.status}`);
        span.dataset.event = event.id;
        span.style.cssText = `left:${position.x + 11}px;top:${position.y}px;height:${Math.max(3, positionAt(event.end) - position.y)}px`;
        stage.append(span);
      }
      if (event.kind === "assignment") {
        const band = node("div", `task-band${event.status === "error" ? " failed" : ""}`);
        band.dataset.event = event.id;
        band.style.cssText = `left:${position.x + 7}px;top:${position.y}px;width:${laneWidth - 14}px;height:${Math.max(cardHeight, positionAt(event.end ?? trace.end) - position.y)}px`;
        band.style.setProperty("--actor-color", colors[actors.get(event.actor).index % colors.length]);
        stage.append(band);
      }
      const card = node("button", `event ${event.kind} ${event.status}`);
      card.type = "button";
      card.dataset.event = event.id;
      card.dataset.actor = event.actor;
      card.style.cssText = `left:${position.x + 17}px;top:${position.y}px;width:${laneWidth - 34}px;height:${cardHeight}px`;
      card.append(node("span", "event-title", event.title));
      const meta = node("span", "event-meta");
      meta.append(node("span", "", time(event.start)), node("span", "", statusText(event)));
      card.append(meta);
      card.title = `${actors.get(event.actor).name} · ${event.title}\n${time(event.start)} · ${statusText(event)}`;
      card.addEventListener("click", () => openDetail(event));
      stage.append(card);
      cards.set(event.id, card);
    }
    drawLinks(width, y + 50);
    applyFilters();
    if (selected) cards.get(selected)?.classList.add("selected");
  }

  function drawLinks(width, height) {
    const namespace = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(namespace, "svg");
    svg.classList.add("handoffs");
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    svg.setAttribute("aria-hidden", "true");
    const defs = document.createElementNS(namespace, "defs");
    const marker = document.createElementNS(namespace, "marker");
    for (const [key, value] of Object.entries({ id: "arrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto-start-reverse" })) marker.setAttribute(key, value);
    const arrow = document.createElementNS(namespace, "path");
    arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
    arrow.style.fill = "#6b96af";
    arrow.style.stroke = "none";
    marker.append(arrow);
    defs.append(marker);
    svg.append(defs);
    for (const link of trace.links) {
      const from = positions.get(link.from);
      const to = positions.get(link.to);
      if (!from || !to) continue;
      const direction = from.x < to.x ? 1 : -1;
      const x1 = from.x + (direction > 0 ? laneWidth - 17 : 17);
      const x2 = to.x + (direction > 0 ? 17 : laneWidth - 17);
      const y1 = from.y + 21;
      const y2 = to.y + 21;
      const path = document.createElementNS(namespace, "path");
      path.setAttribute("d", `M ${x1} ${y1} C ${x1 + 45 * direction} ${y1}, ${x2 - 45 * direction} ${y2}, ${x2} ${y2}`);
      path.setAttribute("marker-end", "url(#arrow)");
      path.dataset.from = link.from;
      path.dataset.to = link.to;
      svg.append(path);
    }
    $("stage").append(svg);
  }

  const searchable = new Map();
  function matches(event) {
    const kind = ["assignment", "lifecycle"].includes(event.kind) ? "task" : event.kind;
    if (!document.querySelector(`[data-kind="${kind}"]`).checked) return false;
    if ($("errors-only").checked && !["error", "warning"].includes(event.status)) return false;
    const query = $("search").value.trim().toLocaleLowerCase();
    if (!query) return true;
    if (!searchable.has(event.id)) {
      const values = [event.title, event.assignment, actors.get(event.actor).name];
      for (const key of ["input", "output", "response", "result", "feedback"]) if (event[key] !== undefined) values.push(trace.blobs[event[key]]);
      for (const id of event.messages || []) values.push(trace.blobs[id]);
      searchable.set(event.id, values.join(" ").toLocaleLowerCase());
    }
    return searchable.get(event.id).includes(query);
  }

  function applyFilters() {
    let count = 0;
    const visible = new Set();
    for (const event of trace.events) {
      const show = matches(event);
      cards.get(event.id).hidden = !show;
      if (show) { count++; visible.add(event.id); }
    }
    for (const band of document.querySelectorAll(".task-band, .activity-span")) band.hidden = !visible.has(band.dataset.event);
    for (const path of document.querySelectorAll(".handoffs path[data-from]")) {
      path.style.display = $("show-links").checked && visible.has(path.dataset.from) && visible.has(path.dataset.to) ? "" : "none";
    }
    $("visible-count").textContent = `${count} / ${trace.events.length} 事件`;
  }

  function block(label, content) {
    const wrapper = node("section", "message");
    wrapper.append(node("div", "message-label", label), node("pre", "", pretty(content)));
    return wrapper;
  }

  function openDetail(event, preferredTab) {
    returnFocus = document.activeElement;
    selected = event.id;
    for (const [id, card] of cards) card.classList.toggle("selected", id === selected);
    $("detail").hidden = false;
    $("detail-title").textContent = `${actors.get(event.actor).name} · ${event.title}`;
    $("detail-time").textContent = `${time(event.start)} → ${event.end === null ? "未记录结束" : time(event.end)} · ${statusText(event)}` + (event.assignment ? `\nAssignment ${event.assignment}` : "") + (event.end_inferred ? " · 就绪时间未记录，以收取时刻为上界" : "");
    const isModel = event.kind === "model";
    const navigation = $("request-nav");
    navigation.hidden = !isModel;
    if (isModel) {
      navigation.replaceChildren();
      for (const request of trace.events.filter((item) => item.kind === "model" && item.actor === event.actor)) {
        const option = node("option", "", `${time(request.start)} · ${request.title}`);
        option.value = request.id;
        navigation.append(option);
      }
      navigation.value = event.id;
    }
    const tabs = isModel ? [["response", "响应"], ["system", "System"], ["user", "User"], ["messages", "完整消息"], ["tools", "工具定义"]] :
      [["input", "输入"], ["output", "输出"], ...(event.result !== undefined ? [["result", "研究交付"]] : []), ...(event.feedback !== undefined ? [["feedback", "校验反馈"]] : [])];
    activeTab = preferredTab || tabs[0][0];
    if (!tabs.some(([key]) => key === activeTab)) activeTab = tabs[0][0];
    $("detail-tabs").replaceChildren();
    for (const [key, label] of tabs) {
      const button = node("button", "", label);
      button.type = "button";
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", String(key === activeTab));
      button.dataset.tab = key;
      button.addEventListener("click", () => {
        activeTab = key;
        for (const tab of $("detail-tabs").children) tab.setAttribute("aria-selected", String(tab.dataset.tab === key));
        renderBody(event);
      });
      $("detail-tabs").append(button);
    }
    renderBody(event);
    $("close-detail").focus({ preventScroll: true });
  }

  function renderBody(event) {
    const body = $("detail-body");
    body.replaceChildren();
    body.scrollTop = 0;
    if (["system", "user", "messages"].includes(activeTab)) {
      if (!event.snapshot) {
        body.append(node("p", "empty", "未保存这次请求的原始快照，无法还原 prompt。"));
        return;
      }
      messages(event).forEach((message, index) => {
        if (activeTab !== "messages" && message.role !== activeTab) return;
        const content = activeTab === "messages" ? message : message.content;
        body.append(block(`${message.role} · 消息 ${index + 1}`, content));
      });
      if (!body.children.length) body.append(node("p", "empty", "这次请求没有该角色的消息。"));
      return;
    }
    const value = blob(event[activeTab]);
    if (value === null) body.append(node("p", "empty", "未保存该项内容。"));
    else body.append(block(({ response: "模型原始响应", tools: "本次请求的工具定义", input: "调用输入", output: "返回结果", result: "任务交付", feedback: "宿主校验回执" })[activeTab], value));
    if (activeTab === "response" && event.status === "error") body.append(block("请求结束记录", blob(event.output)));
  }

  function closeDetail() {
    $("detail").hidden = true;
    cards.get(selected)?.classList.remove("selected");
    selected = null;
    returnFocus?.focus({ preventScroll: true });
  }

  function jumpTo(event) {
    const viewport = $("viewport");
    const position = positions.get(event.id);
    const left = position.x < viewport.scrollLeft + clockWidth || position.x + laneWidth > viewport.scrollLeft + viewport.clientWidth ? Math.max(0, position.x - clockWidth) : viewport.scrollLeft;
    viewport.scrollTo({ top: Math.max(0, position.y - viewport.clientHeight / 3), left });
  }

  $("title").textContent = trace.title;
  const modelCount = trace.events.filter((event) => event.kind === "model").length;
  const toolCount = trace.events.filter((event) => event.kind === "tool").length;
  $("run-info").textContent = `${dateFormatter.format(new Date(trace.start))} · ${time(trace.start)}—${time(trace.end)} · ${trace.timezone} · ${modelCount} 次模型请求 · ${toolCount} 次工具调用`;
  if (trace.warnings.length) {
    $("warnings").hidden = false;
    $("warnings").textContent = trace.warnings.join(" ");
  }
  renderHeadings();
  renderTimeline();
  for (const event of trace.events.filter((item) => item.kind === "assignment")) {
    const option = node("option", "", `${time(event.start)} · ${actors.get(event.actor).name} · ${event.title}`);
    option.value = event.id;
    $("jump").append(option);
  }
  $("jump").addEventListener("change", () => {
    const event = events.get($("jump").value);
    if (event) jumpTo(event);
  });
  for (const input of document.querySelectorAll(".toolbar input")) input.addEventListener("input", applyFilters);
  $("spacing").addEventListener("change", renderTimeline);
  $("request-nav").addEventListener("change", () => openDetail(events.get($("request-nav").value), activeTab));
  $("close-detail").addEventListener("click", closeDetail);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDetail(); });
  $("first-error").addEventListener("click", () => {
    const event = trace.events.find((item) => ["error", "warning"].includes(item.status) && matches(item));
    if (event) { jumpTo(event); openDetail(event); }
  });
  let resizeTimer;
  window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(renderTimeline, 120); });
})();
