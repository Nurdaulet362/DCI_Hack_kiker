'use strict';
const $ = id => document.getElementById(id);
const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = value => value == null ? '—' : new Intl.NumberFormat('ru', {maximumFractionDigits:2}).format(value);
const answerLabel = a => ({yes:'Да', no:'Нет', na:'N/A'}[a] ?? a);
const state = {conversations:[], checklists:[], ai:{}, selected:new Set(), activeId:null,
  conversation:null, evaluation:null, history:[], view:'dialogs', editor:null, job:null,
  pendingAppeal:null, pendingResolve:null, loadVersion:0, pane:"result", editorDirty:false, draftRestored:false, latestJob:null, starting:false, reviewMode:null, checklistId:null};
let toastTimer, pollTimer, pollingId;
const draftKey='cqa-checklist-draft-v1';
const activeStatuses=['queued','running','cancelling'];
function isReview(c) { return !!(c.review_needed || c.score?.gross_error); }
function badgeTone(c, answer) {
  if(answer==='na') return 'neutral';
  if(c.kind==='flag') return answer==='yes'?'danger':'good';
  if(c.kind==='scale') { const value=Number(answer), max=Math.max(...c.levels.map(l=>Number(l.value))); return value<=0?'danger':value<max?'warn':'good'; }
  return answer==='no'?'danger':'good';
}
function updateDraftStatus() {
  $('draftStatus').textContent=state.editorDirty?'Черновик сохранён в этой вкладке · новая версия ещё не создана':'Все изменения сохранены';
  $('discardDraft').classList.toggle('hidden',!state.editorDirty);
}
function rememberDraft() {
  state.editorDirty=true;
  updateDraftStatus();
  try {sessionStorage.setItem(draftKey,JSON.stringify(state.editor));} catch { $('draftStatus').textContent='Черновик хранится только до обновления страницы: хранилище браузера недоступно'; }
}
function restoreEditor() {
  if(state.editor) return;
  try { const draft=JSON.parse(sessionStorage.getItem(draftKey)); if(draft && typeof draft.name==='string' && Array.isArray(draft.criteria) && draft.criteria.every(c=>c && typeof c==='object')) {state.editor=draft;state.editorDirty=true;} } catch {}
  state.editor ||= structuredClone(state.checklists[0]);
}
function setPane(pane) {
  state.pane=pane;
  const detail=$('detail');detail.dataset.pane=pane;
  detail.querySelectorAll('[data-pane]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.pane===pane)));
}
function toast(text) {
  document.querySelectorAll('.dialog-message').forEach(e=>e.remove());
  const dialog=document.querySelector('dialog[open]');
  if(dialog) { const message=document.createElement('p'); message.className='dialog-message error-text'; message.setAttribute('role','alert'); message.textContent=text; dialog.append(message); return; }
  $('toast').textContent=text; $('toast').classList.remove('hidden'); clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').classList.add('hidden'),7000);
}
async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const data = await response.json();
  if (!response.ok) throw Error(data.error || 'Не удалось выполнить запрос.');
  return data;
}
function safeUrl(value) { try { const u=new URL(value); return ['https:','http:'].includes(u.protocol) ? u.href : null; } catch { return null; } }
function download(value, filename) { const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'})); const a=document.createElement('a'); a.href=url; a.download=filename; a.click(); setTimeout(()=>URL.revokeObjectURL(url),500); }
function visibleConversations() { const q=$('search').value.toLocaleLowerCase(), lang=$('language').value, f=$('filter').value;
  return state.conversations.filter(c=>(!lang || c.language===lang) && (!q || `${c.conversation_id} ${c.preview}`.toLocaleLowerCase().includes(q)) && (!f || (f==='new' ? !c.evaluation_id || c.stale : f==='done' ? !!c.evaluation_id && !c.stale : f==='review'?isReview(c):c.score?.gross_error))); }
function renderList() {
  const records=visibleConversations();
  $('visibleCount').textContent=records.length;
  $('selectedCount').textContent=state.selected.size;
  $('batchBtn').disabled=!!state.job || state.starting || !state.selected.size || !state.ai.ready;
  const hiddenSelected=[...state.selected].filter(id=>!records.some(c=>c.conversation_id===id)).length;
  $('selectionNote').textContent=hiddenSelected?`В выборе есть ${hiddenSelected} диалогов вне текущего фильтра. Они тоже будут оценены.`:'';
  $('selectionNote').classList.toggle('hidden',!hiddenSelected);
  $('selectAll').checked=records.length>0 && records.every(c=>state.selected.has(c.conversation_id));
  $('selectAll').indeterminate=records.some(c=>state.selected.has(c.conversation_id)) && !$('selectAll').checked;
  $('conversationList').innerHTML=records.map(c=>`<article class="conversation-row ${c.conversation_id===state.activeId?'active':''}"><input type="checkbox" data-select="${escapeHtml(c.conversation_id)}" aria-label="Выбрать ${escapeHtml(c.conversation_id)}" ${state.selected.has(c.conversation_id)?'checked':''}><button class="row-open" aria-current="${c.conversation_id===state.activeId?'true':'false'}" data-open="${escapeHtml(c.conversation_id)}"><span class="row-top"><strong>${escapeHtml(c.conversation_id)}</strong><span class="badge ${c.stale?'warn':c.score?.gross_error?'danger':c.score?'good':''}">${c.stale?'Обновлён':c.score ? fmt(c.score.total) : 'Не оценён'}</span></span><span class="row-preview">${escapeHtml(c.preview || 'Запрос клиента отсутствует')}</span>${isReview(c)?'<span class="review-hint">Нужна ручная проверка</span>':''}<span class="row-meta"><span class="badge">${escapeHtml(c.language)}</span>${c.channel==='voice call'?'ЗВОНОК':'ЧАТ'} <span>· ${c.messages} реплик</span></span></button></article>`).join('') || '<div class="empty compact"><p>Диалоги не найдены. Измените фильтры или загрузите набор.</p></div>';
}
async function refresh() {
  const data=await api('/api/state');
  Object.assign(state,{conversations:data.conversations,checklists:data.checklists,ai:data.ai});
  if(state.reviewMode===null) state.reviewMode=!!state.ai.review;
  const scored=state.conversations.filter(c=>c.score && !c.stale), numeric=scored.filter(c=>c.score.total!=null && c.checklist_id===state.checklists[0]?.id);
  $('reviewCount').textContent=state.conversations.filter(isReview).length;
  $('count').textContent=state.conversations.length;
  $('evaluated').innerHTML=`${scored.length} <small>/ ${state.conversations.length}</small>`;
  $('average').textContent=numeric.length ? fmt(numeric.reduce((sum,c)=>sum+c.score.total,0)/numeric.length) : '—';
  $('scoreNote').textContent=numeric.length ? `По ${numeric.length} диалогам · версия ${state.checklists.length}` : 'Появится после AI-оценки';
  $('languages').textContent=['RU','KZ','EN'].map(lang=>`${lang} ${state.conversations.filter(c=>c.language===lang).length}`).join(' · ');
  $('aiStatus').textContent=state.ai.ready ? `${state.ai.model} · настроен` : 'AI не подключён';
  $('configBtn').classList.toggle('ready',state.ai.ready);
  $('notice').textContent=state.ai.ready ? 'Анализ '+(state.ai.review?'с повторной проверкой':'в один проход')+' · Точность относительно эталона ещё не измерена.' : 'Диалоги готовы. Для реальной AI-оценки подключите модель в .env после уточнения провайдера. Оценки и баллы пока не рассчитаны.';
  $('configInfo').innerHTML=`<p><strong>Статус:</strong> ${state.ai.ready?'настройки заполнены, доступность проверяется при запуске оценки':'требуются настройки'}<br><strong>Провайдер:</strong> ${escapeHtml(state.ai.provider)}<br><strong>Модель:</strong> ${escapeHtml(state.ai.model || 'не выбрана')}</p>`;
  renderList();
  const job=data.active_job || data.latest_job;
  if(job) {renderJob(job);if(activeStatuses.includes(job.status)) resumePolling(job.id);}
}
function criterionChoices(c) { return (c.kind==='scale' ? c.levels.map(l=>String(l.value)) : ['yes','no']).concat(c.allow_na ? ['na'] : []); }
function renderResult(e) {
  if (!e) return `<div class="empty compact"><span class="empty-icon">◇</span><h2>Оценка ещё не запускалась</h2><p>AI проверит критерии и укажет реплики, на которых основан каждый вывод.</p>${!state.ai.ready?'<button id="connectFromDetail">Подключить AI</button>':''}</div>`;
  const score=e.final_score, a=e.final_assessment;
  const sections=Object.entries(score.sections).filter(([,v])=>v.applicable_weight>0);
  return `<div class="pane-heading"><span>РЕЗУЛЬТАТ ОЦЕНКИ</span><span>${e.appeals.some(x=>x.status==='resolved')?'ПОСЛЕ ПЕРЕСМОТРА':'AI'}</span></div><div class="score-block"><div><span class="score-number">${fmt(score.total)}</span><small> / 100</small></div><div class="score-copy"><strong>${score.total==null?'Нет применимых критериев с весом':'Итоговая оценка'}</strong><p>${score.gross_error?'<span class="badge danger">Грубая ошибка выявлена</span>':'Грубых ошибок '+(score.flags.length?'':'по текущим флагам не выявлено')}</p></div></div><p class="result-summary">${escapeHtml(a.summary)}</p><div class="sections">${sections.map(([k,v])=>`<span class="badge">${escapeHtml(k)}: ${fmt(v.total)}</span>`).join('')}</div>
  ${a.uncertainties?.length?`<div class="notice">Нужна проверка: ${a.uncertainties.map(escapeHtml).join('; ')}</div>`:''}
  ${e.checklist.criteria.map((c,i)=>{ const r=a.criteria[c.id], original=e.assessment.criteria[c.id], points=score.criteria.find(x=>x.id===c.id), open=e.appeals.find(x=>x.criterion_id===c.id && x.status==='open');
    const tone=badgeTone(c,r.answer);
    return `<article class="criterion"><div class="criterion-top"><h3>${String(i+1).padStart(2,'0')} / ${escapeHtml(c.title)}</h3><span class="badge ${tone}">${escapeHtml(answerLabel(r.answer))}${c.kind==='scale'&&r.answer!=='na'?` / ${fmt(Math.max(...c.levels.map(l=>l.value)))}`:''}</span></div><p class="reason">${escapeHtml(r.reason)}</p>${r.reviewer_reason?`<div class="notice">Проверяющий: ${escapeHtml(r.reviewer_reason)}<br>Исходный ответ AI: ${escapeHtml(answerLabel(original.answer))}</div>`:''}<div class="criterion-meta">${c.kind==='flag'?'Отдельный флаг, без баллов':`${fmt(points.points)} из ${fmt(points.effective_weight)} баллов · исходный вес ${fmt(c.weight)}`}</div>
    ${r.evidence.map(v=>`<button class="evidence" data-evidence="${escapeHtml(v.message_id)}"><span>↗ ${escapeHtml(v.message_id)} · ${escapeHtml(({observed:'Действие',absence:'Проверенный контекст',not_applicable:'Основание N/A'})[r.support_type])}</span>«${escapeHtml(v.quote)}»</button>`).join('')}
    <p class="small-note">Проверен участок ${escapeHtml(r.checked_start)}–${escapeHtml(r.checked_end)} · уверенность AI ${Math.round(r.confidence*100)}% (самооценка)</p>
    ${open?`<span class="badge warn">Апелляция на рассмотрении</span> <button class="text-button" data-resolve="${escapeHtml(open.id)}">Рассмотреть</button>`:`<button class="text-button" data-appeal="${escapeHtml(c.id)}">Оспорить оценку</button>`}</article>`; }).join('')}
  <details class="result-provenance"><summary>Модель, время и история проверки</summary><div>${escapeHtml(e.provider)} / ${escapeHtml(e.model)} · ${e.reviewed?'с повторной проверкой':'один проход'}<br>${e.usage.calls} вызовов · ${e.usage.input_tokens+e.usage.output_tokens} токенов · ${fmt(e.usage.seconds)} сек. · стоимость ${e.usage.cost==null?'не задана':fmt(e.usage.cost)+' USD'}<br>Изменено при повторной проверке: ${e.review_changes.length}<br>Исходный итог AI: ${fmt(e.score.total)} · Правила: ${escapeHtml(e.prompt_version)}<br>SHA-256 транскрипта: ${escapeHtml(e.transcript_hash.slice(0,16))}…<br>${escapeHtml(new Date(e.created).toLocaleString('ru'))}</div></details>`;
}
function renderDetail() {
  const current=state.conversation, e=state.evaluation;
  if (!current) return;
  const conv=e?e.conversation:current, audio=safeUrl(conv.audio_url);
  const isOld=!!e && JSON.stringify(conv)!==JSON.stringify(current);
  $('detail').innerHTML=`<div class="detail-heading"><div><span class="badge">${escapeHtml(conv.language)} · ${conv.channel==='voice call'?'Звонок':'Чат'}</span><h2>${escapeHtml(conv.conversation_id)}</h2><p class="small-note">${conv.transcript.length} реплик · ${e?'сохранённая версия транскрипта':'текущий транскрипт'}</p></div><div class="detail-actions"><select id="detailChecklist" aria-label="Чек-лист для новой оценки">${state.checklists.map((c,i)=>`<option value="${escapeHtml(c.id)}" ${c.id===(state.checklistId || state.checklists[0]?.id)?'selected':''}>${escapeHtml(c.name)} · v${state.checklists.length-i}</option>`).join('')}</select><select id="reviewMode" aria-label="Режим оценки"><option value="review" ${state.reviewMode?'selected':''}>С повторной проверкой</option><option value="single" ${!state.reviewMode?'selected':''}>Один проход · быстрее</option></select><button class="primary" id="evaluateBtn" ${!state.ai.ready||state.job||state.starting?'disabled':''}>${e?'Оценить заново':'Запустить оценку'}</button>${e?'<button id="exportBtn">Экспорт JSON ↓</button>':''}</div></div>
  <p class="mode-note">${state.reviewMode?'Два прохода AI.':'Один проход AI без повторного анализа.'} Влияние режима на точность ещё не измерено.</p>
  ${isOld?'<div class="snapshot-warning">Показана сохранённая версия. Текущий транскрипт изменился — запустите новую оценку.</div>':''}
  ${e && e.checklist.id!==state.checklists[0]?.id?'<div class="snapshot-warning">Эта оценка использует предыдущую версию чек-листа. Новые правила применятся при повторной оценке.</div>':''}
  ${state.history.length?`<div class="history-bar">История: <select id="historySelect" aria-label="История оценок"><option value="">Текущий диалог без результата</option>${state.history.map((h,i)=>`<option value="${h.id}" ${e?.id===h.id?'selected':''}>${state.history.length-i} · ${escapeHtml(new Date(h.created).toLocaleString('ru'))}</option>`).join('')}</select></div>`:''}
  <div class="pane-switch" role="group" aria-label="Содержимое карточки"><button data-pane="transcript" aria-pressed="${state.pane==='transcript'}">Диалог</button><button data-pane="result" aria-pressed="${state.pane==='result'}">Оценка</button></div><div class="detail-grid"><div class="transcript-pane"><div class="pane-heading"><span>ТРАНСКРИПТ</span><span>ИСХОДНЫЕ РЕПЛИКИ</span></div>${audio?`<a class="audio-link" href="${escapeHtml(audio)}" target="_blank" rel="noopener noreferrer">↗ Открыть запись звонка</a>`:''}<div class="messages">${conv.transcript.map(m=>`<article class="message ${m.speaker}" id="msg-${m.id}"><div class="message-meta"><span>${({operator:'Оператор',client:'Клиент',bot:'Бот',system:'Система'})[m.speaker]}</span><span>${escapeHtml(m.timestamp)} · ${m.id}</span></div><p>${escapeHtml(m.text)}</p></article>`).join('')}</div></div><div class="result-pane">${renderResult(e)}</div></div>`;
  setPane(state.pane);
}
async function openConversation(id, evaluationId) {
  const version=++state.loadVersion;
  state.activeId=id; renderList();
  const data=await api('/api/conversation?id='+encodeURIComponent(id));
  const eid=evaluationId || data.history[0]?.id;
  const evaluation=eid?await api('/api/evaluation?id='+encodeURIComponent(eid)):null;
  if (version!==state.loadVersion) return;
  Object.assign(state,{conversation:data.conversation, history:data.history, evaluation}); renderDetail();
}
async function evaluateIds(ids) {
  if(state.starting || state.job) return;
  state.starting=true;renderList();
  const checklistId=$('detailChecklist')?.value || state.checklists[0]?.id;
  renderDetail();
  try { const {job_id}=await api('/api/evaluate',{ids,checklist_id:checklistId,review:state.reviewMode});state.job=job_id;resumePolling(job_id); }
  finally {state.starting=false;renderList();renderDetail();}
}
function renderJob(job) {
  state.latestJob=job;state.job=activeStatuses.includes(job.status)?job.id:null;
  $('job').classList.remove('hidden');
  const labels={queued:'В очереди',running:'Анализ выполняется',cancelling:'Остановка после текущего запроса',completed:'Оценка завершена',completed_with_errors:'Есть ошибки',cancelled:'Оценка остановлена',interrupted:'Задача прервана перезапуском'};
  $('jobTitle').textContent=`${labels[job.status] || job.status}: ${job.completed} / ${job.total}`;
  $('jobMessage').textContent=job.message || '';
  $('jobProgress').max=job.total || 1;$('jobProgress').value=job.completed;
  $('cancelJob').classList.toggle('hidden',!state.job);$('cancelJob').disabled=job.status==='cancelling';
  $('retryJob').classList.toggle('hidden',!!state.job || !job.retryable);$('retryJob').textContent=`Повторить оставшиеся (${job.retryable || 0})`;
  $('jobErrors').textContent=(job.errors || []).map(e=>`${e.conversation_id}: ${e.error}`).join('\n');
  $('jobErrors').classList.toggle('hidden',!(job.errors || []).length);
}
function resumePolling(id) {
  if(pollingId===id) return;
  clearTimeout(pollTimer);pollingId=id;pollJob(id);
}
async function pollJob(id) {
  try {
    const job=await api('/api/job?id='+encodeURIComponent(id));
    if(pollingId!==id) return;
    renderJob(job);renderList();
    if(activeStatuses.includes(job.status)) {pollTimer=setTimeout(()=>pollJob(id),1500);return;}
    pollingId=null;await refresh();
    if(state.activeId) await openConversation(state.activeId);
    toast(job.status==='completed'?'Результаты сохранены.':job.message || 'Проверьте состояние задачи.');
  } catch(err) {
    if(pollingId!==id) return;
    $('jobMessage').textContent='Связь с сервером потеряна. Восстанавливаем прогресс…';
    pollTimer=setTimeout(()=>pollJob(id),5000);
  }
}
function showView(view) {
  if(state.view==='checklists' && state.editor) collectEditor();
  state.view=view;
  for (const v of ['dialogs','checklists','reviews']) $(v+'View').classList.toggle('hidden',v!==view);
  document.querySelectorAll('[data-view]').forEach(b=>{b.classList.toggle('active',b.dataset.view===view);if(b.dataset.view===view)b.setAttribute('aria-current','page');else b.removeAttribute('aria-current');});
  $('pageTitle').textContent=({dialogs:'Качество в каждом диалоге.',checklists:'Правила, которым можно доверять.',reviews:'Пересмотр с доказательствами.'})[view];
  $('pageSubtitle').textContent=({dialogs:'Оценка, которую можно проверить по репликам.',checklists:'Изменяйте критерии и применяйте их к новым диалогам.',reviews:'Апелляция, доказательства и финальный результат.'})[view];
  if(view==='checklists') {restoreEditor();renderEditor();}
  if (view==='reviews') renderReviews().catch(e=>toast(e.message));
  if (view==='dialogs') renderDetail();
}
function renderEditor() {
  updateDraftStatus();
  $('checklistName').value=state.editor.name;
  $('criteriaEditor').innerHTML=state.editor.criteria.map((c,i)=>`<article class="editor-card" data-index="${i}"><div class="editor-card-top"><span>КРИТЕРИЙ ${String(i+1).padStart(2,'0')} · ${escapeHtml(c.id)}</span><button type="button" class="text-button" data-remove="${i}">Удалить</button></div><div class="editor-grid"><label class="field">Название<input data-key="title" value="${escapeHtml(c.title)}" required></label><label class="field">Тип<select data-key="kind">${[['binary','Да / Нет'],['scale','Шкала'],['flag','Флаг нарушения']].map(([v,l])=>`<option value="${v}" ${c.kind===v?'selected':''}>${l}</option>`).join('')}</select></label><label class="field">Вес<input data-key="weight" type="number" min="0" step="0.01" value="${escapeHtml(c.weight)}"></label></div><label class="field">Секция<input data-key="section" value="${escapeHtml(c.section)}"></label><label class="field">Правило оценки<textarea data-key="rule" rows="3">${escapeHtml(c.rule)}</textarea></label><label class="editor-na"><input type="checkbox" data-key="allow_na" ${c.allow_na?'checked':''}> Разрешить N/A</label><label class="field">Условие N/A<input data-key="na_rule" value="${escapeHtml(c.na_rule)}"></label><label class="field">Ступени шкалы (балл | описание, одна ступень на строку)<textarea data-key="levels" rows="${c.kind==='scale'?5:2}" placeholder="10 | Полностью выполнено\n0 | Не выполнено">${escapeHtml((c.levels||[]).map(l=>`${l.value} | ${l.description}`).join('\n'))}</textarea></label></article>`).join('');
}
function collectEditor() {
  state.editor.name=$('checklistName').value.trim();
  document.querySelectorAll('.editor-card').forEach(card=>{
    const c=state.editor.criteria[Number(card.dataset.index)];
    card.querySelectorAll('[data-key]').forEach(input=>{
      const key=input.dataset.key;
      if (key==='levels') c.levels=input.value.split('\n').filter(s=>s.trim()).map(line=>{ const [v,...rest]=line.split('|'); return {value:Number(v.trim()),description:rest.join('|').trim()}; });
      else if(key==='weight') c.weight=Number(input.value);
      else if(key==='allow_na') c.allow_na=input.checked;
      else c[key]=input.value.trim();
    });
    if(c.kind==='flag') c.weight=0;
    if(c.kind!=='scale') c.levels=[];
  });
  return {name:state.editor.name,criteria:state.editor.criteria};
}
async function renderReviews() {
  const appeals=await api('/api/appeals');
  $('reviewsList').innerHTML=appeals.map(a=>`<article class="review-card"><div class="review-card-top"><h3>${escapeHtml(a.conversation_id)} / ${escapeHtml(a.criterion_id)}</h3><span class="badge ${a.status==='open'?'warn':'good'}">${a.status==='open'?'На рассмотрении':'Решение сохранено'}</span></div><p>${escapeHtml(a.comment)}</p>${a.status==='resolved'?`<p>Решение: <strong>${escapeHtml(answerLabel(a.answer))}</strong> · ${escapeHtml(a.reason)}</p>`:''}<button data-review-open="${escapeHtml(a.evaluation_id)}" data-conversation="${escapeHtml(a.conversation_id)}">Открыть доказательства ↗</button></article>`).join('') || '<div class="empty"><span class="empty-icon">◇</span><h2>Пока нет апелляций</h2><p>Откройте результат оценки и выберите «Оспорить оценку» у нужного критерия.</p></div>';
}
function openAppeal(cid) { state.pendingAppeal={evaluationId:state.evaluation.id,cid}; $('appealTitle').textContent=state.evaluation.checklist.criteria.find(c=>c.id===cid).title; $('appealComment').value=''; $('appealDialog').showModal(); }
function openResolve(aid) { const e=state.evaluation, a=e.appeals.find(a=>a.id===aid), c=e.checklist.criteria.find(c=>c.id===a.criterion_id); state.pendingResolve=aid; $('resolveContext').textContent=`${c.title}. AI: ${answerLabel(e.assessment.criteria[c.id].answer)}. Комментарий: ${a.comment}`; $('resolveAnswer').innerHTML=criterionChoices(c).map(v=>`<option value="${escapeHtml(v)}" ${v===e.final_assessment.criteria[c.id].answer?'selected':''}>${escapeHtml(answerLabel(v))}</option>`).join(''); $('resolveReason').value=''; $('resolveDialog').showModal(); }
async function guarded(fn) { try { await fn(); } catch(err) { toast(err.message); } }
document.addEventListener('click', event=>guarded(async()=>{
  const b=event.target.closest('button');
  if (!b) return;
  if (b.dataset.close) $(b.dataset.close).close();
  if (b.dataset.view) showView(b.dataset.view);
  if (b.dataset.open) await openConversation(b.dataset.open);
  if(b.dataset.pane) setPane(b.dataset.pane);
  if(b.id==='reviewQueue') {showView('dialogs');$('filter').value='review';renderList();$('dialogsView').scrollIntoView({block:'start'});}
  if(b.id==='cancelJob'&&state.job) {renderJob(await api('/api/job/cancel',{job_id:state.job}));}
  if(b.id==='retryJob'&&state.latestJob) {const {job_id}=await api('/api/job/retry',{job_id:state.latestJob.id});state.job=job_id;resumePolling(job_id);renderList();renderDetail();}
  if(b.id==='discardDraft') {state.editor=structuredClone(state.checklists[0]);state.editorDirty=false;try{sessionStorage.removeItem(draftKey);}catch{}renderEditor();}
  if (b.dataset.evidence) {setPane('transcript'); document.querySelectorAll('.message.highlight').forEach(m=>m.classList.remove('highlight')); const m=$('msg-'+b.dataset.evidence); m?.classList.add('highlight'); m?.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches?'auto':'smooth',block:'center'}); }
  if (b.dataset.appeal) openAppeal(b.dataset.appeal);
  if (b.dataset.resolve) openResolve(b.dataset.resolve);
  if (b.dataset.remove!==undefined) { collectEditor(); state.editor.criteria.splice(Number(b.dataset.remove),1);rememberDraft();renderEditor(); }
  if (b.dataset.reviewOpen) { showView('dialogs'); await openConversation(b.dataset.conversation,b.dataset.reviewOpen); }
  if (b.id==='configBtn'||b.id==='connectFromDetail') $('configDialog').showModal();
  if (b.id==='importBtn') $('importDialog').showModal();
  if (b.id==='evaluateBtn') await evaluateIds([state.activeId]);
  if (b.id==='batchBtn') await evaluateIds([...state.selected]);
  if (b.id==='exportBtn') download(state.evaluation,`cqa-${state.activeId}.json`);
  if (b.id==='newCriterion') { collectEditor(); state.editor.criteria.push({id:'custom_'+crypto.randomUUID().slice(0,8),title:'Новый критерий',section:'Дополнительные критерии',kind:'binary',weight:10,allow_na:false,na_rule:'',rule:'',levels:[]});rememberDraft();renderEditor(); }
  if(b.id==='saveChecklist') {b.disabled=true;try {const checklist=collectEditor();await api('/api/checklists',checklist);await refresh();state.editor=structuredClone(state.checklists[0]);state.editorDirty=false;try{sessionStorage.removeItem(draftKey);}catch{}renderEditor();toast('Версия сохранена. Новые правила применятся при следующей оценке.');}finally{b.disabled=false;}}
}));
document.addEventListener('change',event=>guarded(async()=>{
  const el=event.target;
  if(el.id==='detailChecklist') state.checklistId=el.value;
  if(el.id==='reviewMode') {state.reviewMode=el.value==='review';renderDetail();}
  if (el.dataset.select) { el.checked?state.selected.add(el.dataset.select):state.selected.delete(el.dataset.select); renderList(); }
  if(el.id==='selectAll') { for(const c of visibleConversations()) el.checked?state.selected.add(c.conversation_id):state.selected.delete(c.conversation_id); renderList(); }
  if(['language','filter'].includes(el.id)) renderList();
  if(el.id==='historySelect') { state.evaluation=el.value?await api('/api/evaluation?id='+encodeURIComponent(el.value)):null; renderDetail(); }
  if(el.id==='importFile'&&el.files[0]) { if(el.files[0].size>20*1024*1024) throw Error('Файл превышает 20 МБ.'); const body=JSON.parse(await el.files[0].text()); const r=await api('/api/import',body); await refresh(); if(state.activeId) await openConversation(state.activeId); $('importDialog').close(); el.value=''; toast(`Загружено диалогов: ${r.imported}`); }
}));
$('search').addEventListener('input',renderList);
$('checklistsView').addEventListener('input',event=>{if(event.target.matches('input,textarea,select')){collectEditor();rememberDraft();}});
$('importForm').addEventListener('submit',event=>{event.preventDefault(); guarded(async()=>{ $('fetchBtn').disabled=true; $('fetchBtn').textContent='Загружаю…'; try { const r=await api('/api/fetch',{group_id:$('groupId').value.trim()}); await refresh(); if(state.activeId) await openConversation(state.activeId); $('importDialog').close(); toast(`Загружено диалогов: ${r.imported}`); } finally { $('fetchBtn').disabled=false; $('fetchBtn').textContent='Получить из API'; } });});
$('appealForm').addEventListener('submit',event=>{event.preventDefault(); guarded(async()=>{const p=state.pendingAppeal; await api('/api/appeal',{evaluation_id:p.evaluationId,criterion_id:p.cid,comment:$('appealComment').value}); $('appealDialog').close(); await openConversation(state.activeId,p.evaluationId); toast('Апелляция сохранена.');});});
$('resolveForm').addEventListener('submit',event=>{event.preventDefault(); guarded(async()=>{state.evaluation=await api('/api/resolve',{appeal_id:state.pendingResolve,answer:$('resolveAnswer').value,reason:$('resolveReason').value}); $('resolveDialog').close(); await refresh(); renderDetail(); toast('Решение сохранено, итог пересчитан.');});});
guarded(async()=>{await refresh(); if(state.conversations.length) await openConversation(state.conversations[0].conversation_id);});
