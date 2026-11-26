<script setup lang="ts">
import { computed, ref } from "vue";
import { Check, Code2, Copy, FileCode2 } from "lucide-vue-next";
import {
  tokenizePlainLines,
  tokenizePythonLines,
  type PythonCellsPayload,
  type PythonToken,
} from "../utils/pythonCells";

const props = defineProps<{
  payload: PythonCellsPayload;
  compact?: boolean;
}>();

/** 记录最近一次复制成功的 cell 序号，用于短暂展示反馈。 */
const copiedIndex = ref<number | null>(null);

interface RenderedCell {
  index: number;
  source: string;
  cellId?: string;
  revision?: number;
  label?: string;
  language?: "python" | "plain";
  lines: PythonToken[][];
}

const renderedCells = computed<RenderedCell[]>(() =>
  props.payload.cells.map((cell) => ({
    ...cell,
    lines:
      cell.language === "plain"
        ? tokenizePlainLines(cell.source)
        : tokenizePythonLines(cell.source),
  })),
);

/** 复制单个 cell 的原始 Python 代码到剪贴板。 */
async function copyCell(index: number, source: string) {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(source);
    }
    copiedIndex.value = index;
    setTimeout(() => {
      if (copiedIndex.value === index) copiedIndex.value = null;
    }, 1500);
  } catch {}
}
</script>

<template>
  <div :class="['py-cells-view', { compact }]">
    <div
      v-if="payload.notebookName || payload.cells.length > 1 || payload.afterCellId"
      class="py-cells-meta"
    >
      <span v-if="payload.notebookName" class="py-meta-chip">
        <Code2 :size="11" />{{ payload.notebookName }}.ipynb
      </span>
      <span v-if="payload.cells.length > 1" class="py-meta-count">
        {{ payload.cells.length }} cells
      </span>
      <span v-if="payload.afterCellId" class="py-meta-anchor">
        after {{ payload.afterCellId }}
      </span>
    </div>

    <div
      v-for="cell in renderedCells"
      :key="cell.index"
      class="py-cell-block"
    >
      <div class="py-cell-bar">
        <span class="py-cell-label">
          <template v-if="cell.label">
            <FileCode2 :size="11" />
            <span>{{ cell.label }}</span>
            <code class="py-cell-id">{{ cell.lines.length }} 行</code>
          </template>
          <template v-else>
            Cell {{ cell.index }}
            <code v-if="cell.cellId" class="py-cell-id">{{ cell.cellId }}</code>
          </template>
        </span>
        <button
          type="button"
          class="py-copy-btn"
          :title="copiedIndex === cell.index ? '已复制' : '复制代码'"
          @click="copyCell(cell.index, cell.source)"
        >
          <Check v-if="copiedIndex === cell.index" :size="11" />
          <Copy v-else :size="11" />
          <span>{{ copiedIndex === cell.index ? "已复制" : "复制" }}</span>
        </button>
      </div>

      <pre class="py-code"><code><span
        v-for="(line, lineIdx) in cell.lines"
        :key="lineIdx"
        class="py-line"
      ><span class="py-ln">{{ lineIdx + 1 }}</span><span class="py-line-text"><template
          v-if="line.length"
        ><span
            v-for="(tok, tokIdx) in line"
            :key="tokIdx"
            :class="tok.type === 'plain' ? undefined : 'py-tok-' + tok.type"
          >{{ tok.text }}</span></template><template v-else>&#8203;</template></span></span></code></pre>
    </div>
  </div>
</template>
