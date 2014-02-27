/* bpred.c - branch predictor routines */

/* SimpleScalar(TM) Tool Suite
 * Copyright (C) 1994-2003 by Todd M. Austin, Ph.D. and SimpleScalar, LLC.
 * All Rights Reserved.
 *
 * THIS IS A LEGAL DOCUMENT, BY USING SIMPLESCALAR,
 * YOU ARE AGREEING TO THESE TERMS AND CONDITIONS.
 *
 * No portion of this work may be used by any commercial entity, or for any
 * commercial purpose, without the prior, written permission of SimpleScalar,
 * LLC (info@simplescalar.com). Nonprofit and noncommercial use is permitted
 * as described below.
 *
 * 1. SimpleScalar is provided AS IS, with no warranty of any kind, express
 * or implied. The user of the program accepts full responsibility for the
 * application of the program and the use of any results.
 *
 * 2. Nonprofit and noncommercial use is encouraged. SimpleScalar may be
 * downloaded, compiled, executed, copied, and modified solely for nonprofit,
 * educational, noncommercial research, and noncommercial scholarship
 * purposes provided that this notice in its entirety accompanies all copies.
 * Copies of the modified software can be delivered to persons who use it
 * solely for nonprofit, educational, noncommercial research, and
 * noncommercial scholarship purposes provided that this notice in its
 * entirety accompanies all copies.
 *
 * 3. ALL COMMERCIAL USE, AND ALL USE BY FOR PROFIT ENTITIES, IS EXPRESSLY
 * PROHIBITED WITHOUT A LICENSE FROM SIMPLESCALAR, LLC (info@simplescalar.com).
 *
 * 4. No nonprofit user may place any restrictions on the use of this software,
 * including as modified by the user, by any other authorized user.
 *
 * 5. Noncommercial and nonprofit users may distribute copies of SimpleScalar
 * in compiled or executable form as set forth in Section 2, provided that
 * either: (A) it is accompanied by the corresponding machine-readable source
 * code, or (B) it is accompanied by a written offer, with no time limit, to
 * give anyone a machine-readable copy of the corresponding source code in
 * return for reimbursement of the cost of distribution. This written offer
 * must permit verbatim duplication by anyone, or (C) it is distributed by
 * someone who received only the executable form, and is accompanied by a
 * copy of the written offer of source code.
 *
 * 6. SimpleScalar was developed by Todd M. Austin, Ph.D. The tool suite is
 * currently maintained by SimpleScalar LLC (info@simplescalar.com). US Mail:
 * 2395 Timbercrest Court, Ann Arbor, MI 48105.
 *
 * Copyright (C) 1994-2003 by Todd M. Austin, Ph.D. and SimpleScalar, LLC.
 */


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "host.h"
#include "misc.h"
#include "machine.h"
#include "bpred.h"

/* turn this on to enable the SimpleScalar 2.0 RAS bug */
/* #define RAS_BUG_COMPATIBLE */

static struct bpred_t* bpred_alloc(enum bpred_class class);
static void bpred_alloc_btb(
    struct bpred_t* pred,
    unsigned int btb_sets,
    unsigned int btb_assoc);
static void bpred_alloc_retaddr_stack(
    struct bpred_t* pred,
    unsigned int retstack_size);
static void bpred_alloc_btb_and_retaddr_stack(
    struct bpred_t* pred,
    unsigned int btb_sets,
    unsigned int btb_assoc,
    unsigned int retstack_size);

static struct bpred_dir_t* bpred_dir_alloc(enum bpred_class class);
static void initialize_weak_counters(unsigned char* data, int size, unsigned int nbits);

static int is_stateless(struct bpred_t* pred);

static unsigned char get_bimod_max(struct bpred_t* pred);
static unsigned char get_2lev_max(struct bpred_t* pred);
static unsigned char get_meta_max(struct bpred_t* pred);

static unsigned char get_bimod_halfway(struct bpred_t* pred);
static unsigned char get_2lev_halfway(struct bpred_t* pred);
static unsigned char get_meta_halfway(struct bpred_t* pred);

// Get the primary/secondary predictor's max/halfway val. Requires |pred| for the value, and
// |dir_update_ptr| to determine what the primary/secondary predictor is.
static unsigned char get_dir1_max(struct bpred_t* pred, struct bpred_update_t* dir_update_ptr);
static unsigned char get_dir2_max(struct bpred_t* pred, struct bpred_update_t* dir_update_ptr);
static unsigned char get_dir1_halfway(struct bpred_t* pred, struct bpred_update_t* dir_update_ptr);
static unsigned char get_dir2_halfway(struct bpred_t* pred, struct bpred_update_t* dir_update_ptr);

// branch prediction lookup helper functions

static md_addr_t bpred_lookup_with_btb(
    struct bpred_t* pred,
    md_addr_t baddr,                         /* branch address */
    enum md_opcode op,                       /* opcode of instruction */
    struct bpred_update_t *dir_update_ptr);  /* pred state pointer */

static md_addr_t bpred_lookup_without_btb(
    struct bpred_t* pred,
    enum md_opcode op,                       /* opcode of instruction */
    struct bpred_update_t *dir_update_ptr);  /* pred state pointer */

// branch prediction update helper functions

static void bpred_update_stats(
    struct bpred_t* pred,
    struct bpred_update_t *dir_update_ptr,  /* pred state pointer */
    enum md_opcode op,                      /* opcode of instruction */
    int taken,                              /* non-zero if branch was taken */
    int pred_taken,                         /* non-zero if branch was pred taken */
    int correct);                           /* was earlier addr prediction ok? */

static void bpred_update_ras_bug(
    struct bpred_t* pred,
    enum md_opcode op,     /* opcode of instruction */
    md_addr_t baddr);      /* branch address */

static void bpred_update_l1(
    struct bpred_t* pred,
    enum md_opcode op,     /* opcode of instruction */
    md_addr_t baddr,       /* branch address */
    int taken);

static void bpred_update_btb(
    struct bpred_t* pred,
    enum md_opcode op,     /* opcode of instruction */
    md_addr_t baddr,       /* branch address */
    md_addr_t btarget,     /* resolved branch target */
    int taken,             /* non-zero if branch was taken */
    int correct);          /* was earlier addr prediction ok? */

static void bpred_update_predictors(
    struct bpred_t* pred,
    struct bpred_update_t* dir_update_ptr,  /* pred state pointer */
    int taken);                             /* non-zero if branch was taken */

static void bpred_update_primary_predictor(
    struct bpred_t* pred,
    struct bpred_update_t* dir_update_ptr,  /* pred state pointer */
    int taken);                             /* non-zero if branch was taken */

static void bpred_update_secondary_predictor(
    struct bpred_t* pred,
    struct bpred_update_t* dir_update_ptr,  /* pred state pointer */
    int taken);                             /* non-zero if branch was taken */

static void bpred_update_meta_predictor(
    struct bpred_t* pred,
    struct bpred_update_t* dir_update_ptr,  /* pred state pointer */
    int taken);                             /* non-zero if branch was taken */

struct bpred_t* bpred_create_taken() {
  return bpred_alloc(BPredTaken);
}

struct bpred_t* bpred_create_not_taken() {
  return bpred_alloc(BPredNotTaken);
}

struct bpred_t* bpred_create_smart_static() {
  return bpred_alloc(BPredSmartStatic);
}

struct bpred_t*  /* branch predictory instance */
bpred_create_nbit(
    unsigned int nbits,          /* number of saturating counter bits */
    unsigned int bimod_size,     /* bimod table size */
    unsigned int btb_sets,       /* number of sets in BTB */
    unsigned int btb_assoc,      /* BTB associativity */
    unsigned int retstack_size)  /* num entries in ret-addr stack */
{
  struct bpred_t* pred = bpred_alloc(BPredNbit);
  pred->dirpred.bimod = bpred_dir_create_nbit(bimod_size, nbits);
  bpred_alloc_btb_and_retaddr_stack(pred, btb_sets, btb_assoc, retstack_size);
  return pred;
}

struct bpred_t*  /* branch predictory instance */
bpred_create_2level(
    unsigned int nbits,          /* number of saturating counter bits */
    unsigned int l1size,         /* 2lev l1 table size */
    unsigned int l2size,         /* 2lev l2 table size */
    unsigned int shift_width,    /* history register width */
    unsigned int xor,            /* history xor address flag */
    unsigned int btb_sets,       /* number of sets in BTB */
    unsigned int btb_assoc,      /* BTB associativity */
    unsigned int retstack_size)  /* num entries in ret-addr stack */
{
  struct bpred_t* pred = bpred_alloc(BPred2Level);
  pred->dirpred.twolev = bpred_dir_create_2level(nbits, l1size, l2size, shift_width, xor);
  bpred_alloc_btb_and_retaddr_stack(pred, btb_sets, btb_assoc, retstack_size);
  return pred;
}

struct bpred_t* bpred_create_comb(
    unsigned int bimod_size,     /* bimod table size */
    unsigned int l1size,         /* 2lev l1 table size */
    unsigned int l2size,         /* 2lev l2 table size */
    unsigned int meta_size,      /* meta table size */
    unsigned int shift_width,    /* history register width */
    unsigned int xor,            /* history xor address flag */
    unsigned int btb_sets,       /* number of sets in BTB */
    unsigned int btb_assoc,      /* BTB associativity */
    unsigned int retstack_size)  /* num entries in ret-addr stack */
{
  struct bpred_t* pred = bpred_alloc(BPredComb);
  pred->dirpred.bimod = bpred_dir_create_nbit(bimod_size, 2); // TODO: don't hardcode
  pred->dirpred.twolev = bpred_dir_create_2level(2, l1size, l2size, shift_width, xor); // TODO
  pred->dirpred.meta = bpred_dir_create_nbit(meta_size, 2); // TODO: expose this "2"?
  bpred_alloc_btb_and_retaddr_stack(pred, btb_sets, btb_assoc, retstack_size);
  return pred;
}

// static
struct bpred_t* bpred_alloc(enum bpred_class class) {
  struct bpred_t *pred;
  if (!(pred = calloc(1, sizeof(struct bpred_t))))
    fatal("out of virtual memory");
  pred->class = class;
  return pred;
}

// static
void bpred_alloc_btb(struct bpred_t* pred,
                     unsigned int btb_sets,
                     unsigned int btb_assoc) {
  int i;

  if (!btb_sets || (btb_sets & (btb_sets-1)) != 0)
    fatal("number of BTB sets must be non-zero and a power of two");
  if (!btb_assoc || (btb_assoc & (btb_assoc-1)) != 0)
    fatal("BTB associativity must be non-zero and a power of two");

  if (!(pred->btb = malloc(sizeof(bpred_btb_t))))
    fatal("cannot allocate BTB");

  if (!(pred->btb->btb_data = calloc(btb_sets * btb_assoc, sizeof(struct bpred_btb_ent_t))))
    fatal("cannot allocate BTB data");

  pred->btb->sets = btb_sets;
  pred->btb->assoc = btb_assoc;

  if (pred->btb->assoc > 1) {
    for (i=0; i < pred->btb->assoc * pred->btb->sets; i++) {
      if (i % pred->btb->assoc != pred->btb->assoc - 1)
        pred->btb->btb_data[i].next = &pred->btb->btb_data[i+1];
      else
        pred->btb->btb_data[i].next = NULL;

      if (i % pred->btb->assoc != pred->btb->assoc - 1)
        pred->btb->btb_data[i+1].prev = &pred->btb->btb_data[i];
    }
  }
}

// static
void bpred_alloc_retaddr_stack(struct bpred_t* pred, unsigned int retstack_size) {
  if ((retstack_size & (retstack_size-1)) != 0)
    fatal("Return-address-stack size must be zero or a power of two");
  pred->retstack.size = retstack_size;
  if (retstack_size) {
    if (!(pred->retstack.stack = calloc(retstack_size, sizeof(struct bpred_btb_ent_t))))
      fatal("cannot allocate return-address-stack");
  }
  pred->retstack.tos = retstack_size - 1;
}

// static
void bpred_alloc_btb_and_retaddr_stack(struct bpred_t* pred,        /* branch predictory instance */
                                       unsigned int btb_sets,       /* number of sets in BTB */
                                       unsigned int btb_assoc,      /* BTB associativity */
                                       unsigned int retstack_size)  /* num entries in ret-addr stack */
{
  bpred_alloc_btb(pred, btb_sets, btb_assoc);
  bpred_alloc_retaddr_stack(pred, retstack_size);
}

/* create a branch direction predictor */
struct bpred_dir_t* bpred_dir_create_taken() {
  return bpred_dir_alloc(BPredTaken);
}

struct bpred_dir_t* bpred_dir_create_not_taken() {
  return bpred_dir_alloc(BPredNotTaken);
}

struct bpred_dir_t* bpred_dir_create_smart_static() {
  return bpred_dir_alloc(BPredSmartStatic);
}

struct bpred_dir_t* bpred_dir_create_nbit(
    unsigned int table_size,
    unsigned int nbits)
{
  struct bpred_dir_t* pred_dir = bpred_dir_alloc(BPredNbit);

  if (!table_size || (table_size & (table_size-1)) != 0)
    fatal("%d-bit table size, `%d', must be non-zero and a power of two", nbits, table_size);

  pred_dir->config.bimod.size = table_size;

  if (nbits < 1 || nbits > 8)
    fatal("num bits, `%d', must be between 1 and 8, inclusive", nbits);
  pred_dir->config.bimod.nbits = nbits;

  if (!(pred_dir->config.bimod.table = calloc(table_size, sizeof(unsigned char))))
    fatal("cannot allocate %d-bit storage", nbits);

  // initialize counters to weakly this-or-that
  initialize_weak_counters(pred_dir->config.bimod.table, table_size, nbits);

  return pred_dir;
}

struct bpred_dir_t* bpred_dir_create_2level(
    unsigned int nbits,        /* number of saturating counter bits */
    unsigned int l1size,       /* level-1 table size */
    unsigned int l2size,       /* level-2 table size (if relevant) */
    unsigned int shift_width,  /* history register width */
    unsigned int xor)          /* history xor address flag */
{
  struct bpred_dir_t* pred_dir = bpred_dir_alloc(BPred2Level);

  if (nbits < 1 || nbits > 8)
    fatal("num bits, `%d', must be between 1 and 8, inclusive", nbits);
  pred_dir->config.two.nbits = nbits;

  if (!l1size || (l1size & (l1size-1)) != 0)
    fatal("level-1 size, `%d', must be non-zero and a power of two", l1size);
  pred_dir->config.two.l1size = l1size;

  if (!l2size || (l2size & (l2size-1)) != 0)
    fatal("level-2 size, `%d', must be non-zero and a power of two", l2size);
  pred_dir->config.two.l2size = l2size;

  if (!shift_width || shift_width > 30)
    fatal("shift register width, `%d', must be non-zero, positive, and <30", shift_width);
  pred_dir->config.two.shift_width = shift_width;

  pred_dir->config.two.xor = xor;

  pred_dir->config.two.shiftregs = calloc(l1size, sizeof(int));
  if (!pred_dir->config.two.shiftregs)
    fatal("cannot allocate shift register table");

  pred_dir->config.two.l2table = calloc(l2size, sizeof(unsigned char));
  if (!pred_dir->config.two.l2table)
    fatal("cannot allocate second level table");

  // initialize counters to weakly this-or-that
  initialize_weak_counters(pred_dir->config.two.l2table, l2size, nbits);

  return pred_dir;
}

// static
void initialize_weak_counters(unsigned char* data, int size, unsigned int nbits) {
  int maxval = (1 << nbits) - 1;
  int flipflop = maxval/2;
  int i = 0;

  for (i = 0; i < size; ++i) {
    data[i] = flipflop;
    flipflop = maxval - flipflop;
  }
}

// static
struct bpred_dir_t* bpred_dir_alloc(enum bpred_class class) {
  struct bpred_dir_t *pred_dir;
  if (!(pred_dir = calloc(1, sizeof(struct bpred_dir_t))))
    fatal("out of virtual memory");
  pred_dir->class = class;
  return pred_dir;
}

/* print branch direction predictor configuration */
void print_bpred_dir_config(
  struct bpred_dir_t *pred_dir,  /* branch direction predictor instance */
  char name[],                   /* predictor name */
  FILE *stream)                  /* output stream */
{
  switch (pred_dir->class) {
  case BPred2Level:
    fprintf(stream,
            "pred_dir: %s: 2-lvl: %d l1-sz, %d bits/ent, %s xor, %d l2-sz, direct-mapped\n",
            name,
            pred_dir->config.two.l1size,
            pred_dir->config.two.shift_width,
            pred_dir->config.two.xor ? "" : "no",
            pred_dir->config.two.l2size);
    break;

  case BPredNbit:
    fprintf(stream,
            "pred_dir: %s: %d-bit: %d entries, direct-mapped\n",
            name,
            pred_dir->config.bimod.nbits,
            pred_dir->config.bimod.size);
    break;

  case BPredTaken:
    fprintf(stream, "pred_dir: %s: predict taken\n", name);
    break;

  case BPredNotTaken:
    fprintf(stream, "pred_dir: %s: predict not taken\n", name);
    break;

  case BPredSmartStatic:
    fprintf(stream, "pred_dir: %s: predict taken for backwards branches, not taken for forwards branches\n", name);
    break;

  default:
    panic("bogus branch direction predictor class");
  }
}

/* print branch predictor configuration */
void
print_bpred_config(struct bpred_t *pred,  /* branch predictor instance */
                   FILE *stream)          /* output stream */
{
  switch (pred->class) {
  case BPredComb:
    print_bpred_dir_config(pred->dirpred.bimod, "bimod", stream);
    print_bpred_dir_config(pred->dirpred.twolev, "2lev", stream);
    print_bpred_dir_config(pred->dirpred.meta, "meta", stream);

    fprintf(stream, "btb: %d sets x %d associativity", pred->btb->sets, pred->btb->assoc);
    fprintf(stream, "ret_stack: %d entries", pred->retstack.size);
    break;

  case BPred2Level:
    print_bpred_dir_config(pred->dirpred.twolev, "2lev", stream);

    fprintf(stream, "btb: %d sets x %d associativity", pred->btb->sets, pred->btb->assoc);
    fprintf(stream, "ret_stack: %d entries", pred->retstack.size);
    break;

  case BPredNbit:
    print_bpred_dir_config(pred->dirpred.bimod, "bimod", stream);

    fprintf(stream, "btb: %d sets x %d associativity", pred->btb->sets, pred->btb->assoc);
    fprintf(stream, "ret_stack: %d entries", pred->retstack.size);
    break;

  case BPredTaken:
    print_bpred_dir_config(pred->dirpred.bimod, "taken", stream);
    break;

  case BPredNotTaken:
    print_bpred_dir_config(pred->dirpred.bimod, "nottaken", stream);
    break;

  case BPredSmartStatic:
    print_bpred_dir_config(pred->dirpred.bimod, "smartstatic", stream);
    break;

  default:
    panic("bogus branch predictor class");
  }
}

/* print predictor stats */
void
print_bpred_stats(struct bpred_t *pred,  /* branch predictor instance */
            FILE *stream)          /* output stream */
{
  fprintf(stream, "pred: addr-prediction rate = %f\n",
    (double)pred->addr_hits/(double)(pred->addr_hits+pred->misses));
  fprintf(stream, "pred: dir-prediction rate = %f\n",
    (double)pred->dir_hits/(double)(pred->dir_hits+pred->misses));
}

/* register branch predictor stats */
void
bpred_reg_stats(struct bpred_t *pred,  /* branch predictor instance */
    struct stat_sdb_t *sdb)  /* stats database */
{
  char buf[512], buf1[512], *name;

  /* get a name for this predictor */
  switch (pred->class)
    {
    case BPredComb:
      name = "bpred_comb";
      break;
    case BPred2Level:
      name = "bpred_2lev";
      break;
    case BPredNbit:
      name = "bpred_bimod";
      break;
    case BPredTaken:
      name = "bpred_taken";
      break;
    case BPredNotTaken:
      name = "bpred_nottaken";
      break;
    case BPredSmartStatic:
      name = "bpred_smartstatic";
      break;
    default:
      panic("bogus branch predictor class");
    }

  sprintf(buf, "%s.lookups", name);
  stat_reg_counter(sdb,
                   buf,
                   "total number of bpred lookups",
                   &pred->lookups,
                   0,
                   NULL);
  sprintf(buf, "%s.updates", name);
  sprintf(buf1, "%s.dir_hits + %s.misses", name, name);
  stat_reg_formula(sdb,
                   buf,
                   "total number of updates",
                   buf1,
                   "%12.0f");
  sprintf(buf, "%s.addr_hits", name);
  stat_reg_counter(sdb,
                   buf,
                   "total number of address-predicted hits",
                   &pred->addr_hits,
                   0,
                   NULL);
  sprintf(buf, "%s.dir_hits", name);
  stat_reg_counter(sdb,
                   buf,
                   "total number of direction-predicted hits (includes addr-hits)",
                   &pred->dir_hits,
                   0,
                   NULL);
  if (pred->class == BPredComb) {
    sprintf(buf, "%s.used_bimod", name);
    stat_reg_counter(sdb, buf,
         "total number of bimodal predictions used",
         &pred->used_bimod, 0, NULL);
    sprintf(buf, "%s.used_2lev", name);
    stat_reg_counter(sdb, buf,
         "total number of 2-level predictions used",
         &pred->used_2lev, 0, NULL);
  }
  sprintf(buf, "%s.misses", name);
  stat_reg_counter(sdb, buf, "total number of misses", &pred->misses, 0, NULL);
  sprintf(buf, "%s.jr_hits", name);
  stat_reg_counter(sdb, buf,
       "total number of address-predicted hits for JR's",
       &pred->jr_hits, 0, NULL);
  sprintf(buf, "%s.jr_seen", name);
  stat_reg_counter(sdb, buf,
       "total number of JR's seen",
       &pred->jr_seen, 0, NULL);
  sprintf(buf, "%s.jr_non_ras_hits.PP", name);
  stat_reg_counter(sdb, buf,
       "total number of address-predicted hits for non-RAS JR's",
       &pred->jr_non_ras_hits, 0, NULL);
  sprintf(buf, "%s.jr_non_ras_seen.PP", name);
  stat_reg_counter(sdb, buf,
       "total number of non-RAS JR's seen",
       &pred->jr_non_ras_seen, 0, NULL);
  sprintf(buf, "%s.bpred_addr_rate", name);
  sprintf(buf1, "%s.addr_hits / %s.updates", name, name);
  stat_reg_formula(sdb, buf,
       "branch address-prediction rate (i.e., addr-hits/updates)",
       buf1, "%9.4f");
  sprintf(buf, "%s.bpred_dir_rate", name);
  sprintf(buf1, "%s.dir_hits / %s.updates", name, name);
  stat_reg_formula(sdb, buf,
      "branch direction-prediction rate (i.e., all-hits/updates)",
      buf1, "%9.4f");
  sprintf(buf, "%s.bpred_jr_rate", name);
  sprintf(buf1, "%s.jr_hits / %s.jr_seen", name, name);
  stat_reg_formula(sdb, buf,
      "JR address-prediction rate (i.e., JR addr-hits/JRs seen)",
      buf1, "%9.4f");
  sprintf(buf, "%s.bpred_jr_non_ras_rate.PP", name);
  sprintf(buf1, "%s.jr_non_ras_hits.PP / %s.jr_non_ras_seen.PP", name, name);
  stat_reg_formula(sdb, buf,
       "non-RAS JR addr-pred rate (ie, non-RAS JR hits/JRs seen)",
       buf1, "%9.4f");
  sprintf(buf, "%s.retstack_pushes", name);
  stat_reg_counter(sdb, buf,
       "total number of address pushed onto ret-addr stack",
       &pred->retstack_pushes, 0, NULL);
  sprintf(buf, "%s.retstack_pops", name);
  stat_reg_counter(sdb, buf,
       "total number of address popped off of ret-addr stack",
       &pred->retstack_pops, 0, NULL);
  sprintf(buf, "%s.used_ras.PP", name);
  stat_reg_counter(sdb, buf,
       "total number of RAS predictions used",
       &pred->used_ras, 0, NULL);
  sprintf(buf, "%s.ras_hits.PP", name);
  stat_reg_counter(sdb, buf,
       "total number of RAS hits",
       &pred->ras_hits, 0, NULL);
  sprintf(buf, "%s.ras_rate.PP", name);
  sprintf(buf1, "%s.ras_hits.PP / %s.used_ras.PP", name, name);
  stat_reg_formula(sdb, buf,
       "RAS prediction rate (i.e., RAS hits/used RAS)",
       buf1, "%9.4f");
}

void
bpred_after_priming(struct bpred_t *bpred)
{
  if (bpred == NULL)
    return;

  bpred->lookups = 0;
  bpred->addr_hits = 0;
  bpred->dir_hits = 0;
  bpred->used_ras = 0;
  bpred->used_bimod = 0;
  bpred->used_2lev = 0;
  bpred->jr_hits = 0;
  bpred->jr_seen = 0;
  bpred->misses = 0;
  bpred->retstack_pops = 0;
  bpred->retstack_pushes = 0;
  bpred->ras_hits = 0;
}

#define BIMOD_HASH(PRED, ADDR)            \
  ((((ADDR) >> 19) ^ ((ADDR) >> MD_BR_SHIFT)) & ((PRED)->config.bimod.size-1))
    /* was: ((baddr >> 16) ^ baddr) & (pred->dirpred.bimod.size-1) */

/* predicts a branch direction */
unsigned char *            /* pointer to counter */
bpred_dir_lookup(
    struct bpred_dir_t *pred_dir,   /* branch dir predictor inst */
    md_addr_t baddr)                /* branch address */
{
  unsigned char *p = NULL;

  /* Except for jumps, get a pointer to direction-prediction bits */
  switch (pred_dir->class) {
    case BPred2Level:
      {
        int l1index, l2index;
        int temp1, temp2, temp3;

        /* traverse 2-level tables */
        l1index = (baddr >> MD_BR_SHIFT) & (pred_dir->config.two.l1size - 1);
        l2index = pred_dir->config.two.shiftregs[l1index];
        if (pred_dir->config.two.xor) {
          /* this L2 index computation is more "compatible" to McFarling's
           * verison of it, i.e., if the PC xor address component is only
           * part of the index, take the lower order address bits for the
           * other part of the index, rather than the higher order ones
           */
          temp1 = l2index ^ (baddr >> MD_BR_SHIFT);
          temp2 =  (1 << pred_dir->config.two.shift_width) - 1;
          temp3 = (baddr >> MD_BR_SHIFT) << pred_dir->config.two.shift_width;
          l2index = (temp1 & temp2) | temp3;

          // l2index = l2index ^ (baddr >> MD_BR_SHIFT);
        } else {
          l2index = l2index | ((baddr >> MD_BR_SHIFT) << pred_dir->config.two.shift_width);
        }

        l2index = l2index & (pred_dir->config.two.l2size - 1);

        /* get a pointer to prediction state information */
        p = &pred_dir->config.two.l2table[l2index];
      }
      break;
    case BPredNbit:
      p = &pred_dir->config.bimod.table[BIMOD_HASH(pred_dir, baddr)];
      break;
    case BPredTaken:
    case BPredNotTaken:
    case BPredSmartStatic:
      break;
    default:
      panic("bogus branch direction predictor class");
    }

  return (unsigned char *)p;
}

/* probe a predictor for a next fetch address, the predictor is probed
   with branch address BADDR, the branch target is BTARGET (used for
   static predictors), and OP is the instruction opcode (used to simulate
   predecode bits; a pointer to the predictor state entry (or null for jumps)
   is returned in *DIR_UPDATE_PTR (used for updating predictor state),
   and the non-speculative top-of-stack is returned in stack_recover_idx
   (used for recovering ret-addr stack after mis-predict).  */
md_addr_t  /* predicted branch target addr */
bpred_lookup(
    struct bpred_t* pred,                   /* branch predictor instance */
    md_addr_t baddr,                        /* branch address */
    md_addr_t btarget,                      /* branch target if taken */
    enum md_opcode op,                      /* opcode of instruction */
    int is_call,                            /* non-zero if inst is fn call */
    int is_return,                          /* non-zero if inst is fn return */
    struct bpred_update_t *dir_update_ptr,  /* pred state pointer */
    int *stack_recover_idx)                 /* Non-speculative top-of-stack; used on mispredict recovery */
{
  if (!dir_update_ptr)
    panic("no bpred update record");

  /* if this is not a branch, return not-taken */
  if (!(MD_OP_FLAGS(op) & F_CTRL))
    return 0;

  pred->lookups++;

  dir_update_ptr->dir.ras = FALSE;
  dir_update_ptr->pdir1 = NULL;
  dir_update_ptr->pdir2 = NULL;
  dir_update_ptr->pmeta = NULL;

  /* Except for jumps, get a pointer to direction-prediction bits */
  switch (pred->class) {
    case BPredComb:
      if (!is_unconditional_control_op(op)) {
        unsigned char* bimod  = bpred_dir_lookup(pred->dirpred.bimod, baddr);
        unsigned char* twolev = bpred_dir_lookup(pred->dirpred.twolev, baddr);
        unsigned char* meta   = bpred_dir_lookup(pred->dirpred.meta, baddr);

        dir_update_ptr->pmeta = meta;

        dir_update_ptr->dir.meta   = *meta >= get_meta_halfway(pred);
        dir_update_ptr->dir.bimod  = *bimod >= get_bimod_halfway(pred);
        dir_update_ptr->dir.twolev = *twolev >= get_2lev_halfway(pred);

        if (*meta >= get_meta_halfway(pred)) {
          dir_update_ptr->pdir1 = twolev;
          dir_update_ptr->pdir2 = bimod;

          dir_update_ptr->dir1_class = BPred2Level;
        } else {
          dir_update_ptr->pdir1 = bimod;
          dir_update_ptr->pdir2 = twolev;

          dir_update_ptr->dir1_class = BPredNbit;
        }
      }
      break;
    case BPred2Level:
      if (!is_unconditional_control_op(op)) {
        dir_update_ptr->pdir1 = bpred_dir_lookup(pred->dirpred.twolev, baddr);
        dir_update_ptr->dir1_class = BPred2Level;
      }
      break;
    case BPredNbit:
      if (!is_unconditional_control_op(op)) {
        dir_update_ptr->pdir1 = bpred_dir_lookup(pred->dirpred.bimod, baddr);
        dir_update_ptr->dir1_class = BPredNbit;
      }
      break;
    case BPredTaken:
      return btarget;
    case BPredNotTaken:
      if (!is_unconditional_control_op(op))
        return baddr + sizeof(md_inst_t);
      else
        return btarget;
    case BPredSmartStatic:
      /* predict backwards branches will be taken */
      if (btarget < baddr) {
        return btarget;
      }
      /* predict forwards branches will not (duplicate above logic) */
      else {
        if (!is_unconditional_control_op(op))
          return baddr + sizeof(md_inst_t);
        else
          return btarget;
      }
    default:
      panic("bogus predictor class");
  }

  /*
   * We have a stateful predictor, and have gotten a pointer into the
   * direction predictor (except for jumps, for which the ptr is null)
   */

  /* record pre-pop TOS; if this branch is executed speculatively
   * and is squashed, we'll restore the TOS and hope the data
   * wasn't corrupted in the meantime. */
  if (pred->retstack.size)
    *stack_recover_idx = pred->retstack.tos;
  else
    *stack_recover_idx = 0;

  /* if this is a return, pop return-address stack */
  if (is_return && pred->retstack.size) {
    md_addr_t target = pred->retstack.stack[pred->retstack.tos].target;
    pred->retstack.tos =
        (pred->retstack.tos + pred->retstack.size - 1) % pred->retstack.size;
    pred->retstack_pops++;
    dir_update_ptr->dir.ras = TRUE; /* using RAS here */
    return target;
  }

#ifndef RAS_BUG_COMPATIBLE
  /* if function call, push return-address onto return-address stack */
  if (is_call && pred->retstack.size) {
    pred->retstack.tos = (pred->retstack.tos + 1)% pred->retstack.size;
    pred->retstack.stack[pred->retstack.tos].target = baddr + sizeof(md_inst_t);
    pred->retstack_pushes++;
  }
#endif /* !RAS_BUG_COMPATIBLE */

  if (pred->btb)
    return bpred_lookup_with_btb(pred, baddr, op, dir_update_ptr);
  else
    return bpred_lookup_without_btb(pred, op, dir_update_ptr);
}

md_addr_t bpred_lookup_with_btb(
    struct bpred_t* pred,
    md_addr_t baddr,                        /* branch address */
    enum md_opcode op,                      /* opcode of instruction */
    struct bpred_update_t *dir_update_ptr)  /* pred state pointer */
{
  struct bpred_btb_ent_t *pbtb_data = NULL;
  int index, i;

  /* not a return. Get a pointer into the BTB */
  index = (baddr >> MD_BR_SHIFT) & (pred->btb->sets - 1);

  if (pred->btb->assoc > 1) {
    index *= pred->btb->assoc;

    /* Now we know the set; look for a PC match */
    for (i = index; i < (index+pred->btb->assoc); i++) {
      if (pred->btb->btb_data[i].addr == baddr) {
        /* match */
        pbtb_data = &pred->btb->btb_data[i];
        break;
      }
    }
  } else {
    pbtb_data = &pred->btb->btb_data[index];
    if (pbtb_data->addr != baddr)
      pbtb_data = NULL;
  }

  /*
   * We now also have a pointer into the BTB for a hit, or NULL otherwise
   */

  /* if this is a jump, ignore predicted direction; we know it's taken. */
  if (is_unconditional_control_op(op))
    return pbtb_data ? pbtb_data->target : TAKEN;

  /* otherwise we have a conditional branch */
  unsigned char direction_val = *(dir_update_ptr->pdir1);
  unsigned char direction_halfway = get_dir1_halfway(pred, dir_update_ptr);

  // if BTB miss, just return a predicted direction (1 or 0). otherwise, return
  // the target from the BTB hit if it's a predicted-taken branch.
  md_addr_t taken_target = pbtb_data == NULL ? TAKEN : pbtb_data->target;

  return direction_val >= direction_halfway ? taken_target : NOT_TAKEN;
}

md_addr_t bpred_lookup_without_btb(
    struct bpred_t* pred,
    enum md_opcode op,                      /* opcode of instruction */
    struct bpred_update_t *dir_update_ptr)  /* pred state pointer */
{
  /* if this is a jump, ignore predicted direction; we know it's taken. */
  if (is_unconditional_control_op(op))
    return TAKEN;

  /* otherwise we have a conditional branch */
  unsigned char direction_val = *(dir_update_ptr->pdir1);
  unsigned char direction_halfway = get_dir1_halfway(pred, dir_update_ptr);
  return direction_val >= direction_halfway ? TAKEN : NOT_TAKEN;
}

/* Speculative execution can corrupt the ret-addr stack.  So for each
 * lookup we return the top-of-stack (TOS) at that point; a mispredicted
 * branch, as part of its recovery, restores the TOS using this value --
 * hopefully this uncorrupts the stack. */
void
bpred_recover(struct bpred_t *pred,  /* branch predictor instance */
        md_addr_t baddr,    /* branch address */
        int stack_recover_idx)  /* Non-speculative top-of-stack;
           * used on mispredict recovery */
{
  if (pred == NULL)
    return;

  pred->retstack.tos = stack_recover_idx;
}

/* update the branch predictor, only useful for stateful predictors; updates
 * entry for instruction type OP at address BADDR.  BTB only gets updated
 * for branches which are taken.  Inst was determined to jump to
 * address BTARGET and was taken if TAKEN is non-zero.  Predictor
 * statistics are updated with result of prediction, indicated by CORRECT and
 * PRED_TAKEN, predictor state to be updated is indicated by *DIR_UPDATE_PTR
 * (may be NULL for jumps, which shouldn't modify state bits).  Note if
 * bpred_update is done speculatively, branch-prediction may get polluted.
 */
void bpred_update(
    struct bpred_t *pred,                   /* branch predictor instance */
    md_addr_t baddr,                        /* branch address */
    md_addr_t btarget,                      /* resolved branch target */
    int taken,                              /* non-zero if branch was taken */
    int pred_taken,                         /* non-zero if branch was pred taken */
    int correct,                            /* was earlier addr prediction ok? */
    enum md_opcode op,                      /* opcode of instruction */
    struct bpred_update_t *dir_update_ptr)  /* pred state pointer */
{
  // don't change bpred state for non-branch instructions or if this is a stateless predictor
  if (!(MD_OP_FLAGS(op) & F_CTRL))
    return;

  /* Have a branch here */

  bpred_update_stats(pred, dir_update_ptr, op, taken, pred_taken, correct);

  if (is_stateless(pred))
    return;

  /*
   * Now we know the branch didn't use the ret-addr stack, and that this
   * is a stateful predictor
   */

  bpred_update_ras_bug(pred, op, baddr); // Wtf is this?

  /* update L1 table if appropriate */
  /* L1 table is updated unconditionally for combining predictor too */
  bpred_update_l1(pred, op, baddr, taken);

  if (pred->btb)
    bpred_update_btb(pred, op, baddr, btarget, taken, correct);

  bpred_update_predictors(pred, dir_update_ptr, taken);
}

void bpred_update_stats(
    struct bpred_t* pred,
    struct bpred_update_t *dir_update_ptr,  /* pred state pointer */
    enum md_opcode op,                      /* opcode of instruction */
    int taken,                              /* non-zero if branch was taken */
    int pred_taken,                         /* non-zero if branch was pred taken */
    int correct)                            /* was earlier addr prediction ok? */
{
  if (correct)
    pred->addr_hits++;

  if (!!pred_taken == !!taken)
    pred->dir_hits++;
  else
    pred->misses++;

  if (dir_update_ptr->dir.ras) {
    pred->used_ras++;
    if (correct)
      pred->ras_hits++;
  } else if ((MD_OP_FLAGS(op) & (F_CTRL|F_COND)) == (F_CTRL|F_COND)) {
    if (dir_update_ptr->dir.meta)
      pred->used_2lev++;
    else
      pred->used_bimod++;
  }

  /* keep stats about JR's; also, but don't change any bpred state for JR's
   * which are returns unless there's no retstack */
  if (MD_IS_INDIR(op)) {
    pred->jr_seen++;
    if (correct)
      pred->jr_hits++;

    if (!dir_update_ptr->dir.ras) {
      pred->jr_non_ras_seen++;
      if (correct)
        pred->jr_non_ras_hits++;
    } else {
      /* return that used the ret-addr stack; no further work to do */
      return;
    }
  }
}

void bpred_update_ras_bug(
    struct bpred_t* pred,
    enum md_opcode op,     /* opcode of instruction */
    md_addr_t baddr)       /* branch address */
{
#ifdef RAS_BUG_COMPATIBLE
  /* if function call, push return-address onto return-address stack */
  if (MD_IS_CALL(op) && pred->retstack.size) {
    pred->retstack.tos = (pred->retstack.tos + 1) % pred->retstack.size;
    pred->retstack.stack[pred->retstack.tos].target = baddr + sizeof(md_inst_t);
    pred->retstack_pushes++;
  }
#endif
}

void bpred_update_l1(
    struct bpred_t* pred,
    enum md_opcode op,     /* opcode of instruction */
    md_addr_t baddr,       /* branch address */
    int taken)             /* non-zero if branch was taken */
{
  if (!is_unconditional_control_op(op) &&
      (pred->class == BPred2Level || pred->class == BPredComb)) {
    int l1index, shift_reg;

    /* also update appropriate L1 history register */
    l1index = (baddr >> MD_BR_SHIFT) & (pred->dirpred.twolev->config.two.l1size - 1);
    shift_reg = (pred->dirpred.twolev->config.two.shiftregs[l1index] << 1) | (!!taken);
    pred->dirpred.twolev->config.two.shiftregs[l1index] =
        shift_reg & ((1 << pred->dirpred.twolev->config.two.shift_width) - 1);
  }
}

void bpred_update_btb(
    struct bpred_t* pred,
    enum md_opcode op,     /* opcode of instruction */
    md_addr_t baddr,       /* branch address */
    md_addr_t btarget,     /* resolved branch target */
    int taken,             /* non-zero if branch was taken */
    int correct)           /* was earlier addr prediction ok? */
{
  struct bpred_btb_ent_t *pbtb_data = NULL;
  struct bpred_btb_ent_t *lruhead = NULL, *lruitem = NULL;
  int index, i;

  /* find BTB entry if it's a taken branch (don't allocate for non-taken) */
  if (taken) {
    index = (baddr >> MD_BR_SHIFT) & (pred->btb->sets - 1);
    if (pred->btb->assoc > 1) {
      index *= pred->btb->assoc;

      /* Now we know the set; look for a PC match; also identify MRU and LRU items */
      for (i = index; i < (index+pred->btb->assoc) ; i++) {
        if (pred->btb->btb_data[i].addr == baddr) {
          /* match */
          assert(!pbtb_data);
          pbtb_data = &pred->btb->btb_data[i];
        }

        dassert(pred->btb->btb_data[i].prev != pred->btb->btb_data[i].next);

        if (pred->btb->btb_data[i].prev == NULL) {
          /* this is the head of the lru list, ie current MRU item */
          dassert(lruhead == NULL);
          lruhead = &pred->btb->btb_data[i];
        }

        if (pred->btb->btb_data[i].next == NULL) {
          /* this is the tail of the lru list, ie the LRU item */
          dassert(lruitem == NULL);
          lruitem = &pred->btb->btb_data[i];
        }
      }

      dassert(lruhead && lruitem);

      if (!pbtb_data) {
        /* missed in BTB; choose the LRU item in this set as the victim */
        pbtb_data = lruitem;
      } else {
        /* else hit, and pbtb_data points to matching BTB entry */
      }

      /* Update LRU state: selected item, whether selected because it
       * matched or because it was LRU and selected as a victim, becomes
       * MRU */
      if (pbtb_data != lruhead) {
        /* this splices out the matched entry... */
        if (pbtb_data->prev)
          pbtb_data->prev->next = pbtb_data->next;
        if (pbtb_data->next)
          pbtb_data->next->prev = pbtb_data->prev;

        /* ...and this puts the matched entry at the head of the list */
        pbtb_data->next = lruhead;
        pbtb_data->prev = NULL;
        lruhead->prev = pbtb_data;
        dassert(pbtb_data->prev || pbtb_data->next);
        dassert(pbtb_data->prev != pbtb_data->next);
      } else {
        /* else pbtb_data is already MRU item; do nothing */
      }
    } else {
      pbtb_data = &pred->btb->btb_data[index];
    }
  }

  /* Now 'pbtb_data' is a possibly null pointer into the BTB (either to a
   * matched-on entry or a victim which was LRU in its set)
   */

  /* update BTB (but only for taken branches) */
  if (pbtb_data) {
    /* update current information */
    dassert(taken);

    if (pbtb_data->addr == baddr) {
      if (!correct)
        pbtb_data->target = btarget;
    } else {
      /* enter a new branch in the table */
      pbtb_data->addr = baddr;
      pbtb_data->op = op;
      pbtb_data->target = btarget;
    }
  }
}

void bpred_update_predictors(
    struct bpred_t* pred,
    struct bpred_update_t* dir_update_ptr,  /* pred state pointer */
    int taken)                              /* non-zero if branch was taken */
{
  if (dir_update_ptr->pdir1)
    bpred_update_primary_predictor(pred, dir_update_ptr, taken);
  if (dir_update_ptr->pdir2)
    bpred_update_secondary_predictor(pred, dir_update_ptr, taken);
  if (dir_update_ptr->pmeta)
    bpred_update_meta_predictor(pred, dir_update_ptr, taken);
}

void bpred_update_primary_predictor(
    struct bpred_t* pred,
    struct bpred_update_t* dir_update_ptr,  /* pred state pointer */
    int taken)                              /* non-zero if branch was taken */
{
  if (taken) {
    if (*dir_update_ptr->pdir1 < get_dir1_max(pred, dir_update_ptr))
      ++*dir_update_ptr->pdir1;
  } else {
    if (*dir_update_ptr->pdir1 > 0)
      --*dir_update_ptr->pdir1;
  }
}

void bpred_update_secondary_predictor(
    struct bpred_t* pred,
    struct bpred_update_t* dir_update_ptr,  /* pred state pointer */
    int taken)                              /* non-zero if branch was taken */
{
  if (taken) {
    if (*dir_update_ptr->pdir2 < get_dir2_max(pred, dir_update_ptr))
      ++*dir_update_ptr->pdir2;
  } else { /* not taken */
    if (*dir_update_ptr->pdir2 > 0)
      --*dir_update_ptr->pdir2;
  }
}

void bpred_update_meta_predictor(
    struct bpred_t* pred,
    struct bpred_update_t* dir_update_ptr,  /* pred state pointer */
    int taken)                              /* non-zero if branch was taken */
{
  if (dir_update_ptr->dir.bimod != dir_update_ptr->dir.twolev) {
    /* we only update meta predictor if directions were different */
    if (dir_update_ptr->dir.twolev == (unsigned int)taken) {
      /* 2-level predictor was correct */
      if (*dir_update_ptr->pmeta < get_meta_max(pred))
        ++*dir_update_ptr->pmeta;
    } else {
      /* bimodal predictor was correct */
      if (*dir_update_ptr->pmeta > 0)
        --*dir_update_ptr->pmeta;
    }
  }
}

// if bits == 3, returns 7
// static
unsigned char get_bimod_max(struct bpred_t* pred) {
  return (1 << pred->dirpred.bimod->config.bimod.nbits) - 1;
}

// if bits == 3, returns 4
// static
unsigned char get_bimod_halfway(struct bpred_t* pred) {
  return 1 << (pred->dirpred.bimod->config.bimod.nbits - 1);
}

// if bits == 3, returns 7
// static
unsigned char get_2lev_max(struct bpred_t* pred) {
  return (1 << pred->dirpred.twolev->config.two.nbits) - 1;
}

// if bits == 3, returns 4
// static
unsigned char get_2lev_halfway(struct bpred_t* pred) {
  return 1 << (pred->dirpred.twolev->config.two.nbits - 1);
}

// if bits == 3, returns 7
// static
unsigned char get_meta_max(struct bpred_t* pred) {
  // TODO: This assumes the meta predictor is bimodal.
  // This assumption is probably made all over the place, anyway.
  return (1 << pred->dirpred.meta->config.bimod.nbits) - 1;
}

// if bits == 3, returns 4
// static
unsigned char get_meta_halfway(struct bpred_t* pred) {
  // TODO: This assumes the meta predictor is bimodal.
  // This assumption is probably made all over the place, anyway.
  return 1 << (pred->dirpred.meta->config.bimod.nbits - 1);
}

/*
 * Get the primary predictor's max val. Requires |pred| for the value, and
 * |dir_update_ptr| to determine what the primary predictor is.
 *
 * Assumes dir_update_ptr->dir1 is non-null.
 *
 * static
 */
unsigned char get_dir1_max(struct bpred_t* pred, struct bpred_update_t* dir_update_ptr) {
  switch (dir_update_ptr->dir1_class) {
  case BPredNbit:
    return get_bimod_max(pred);
  case BPred2Level:
    return get_2lev_max(pred);
  default:
    fatal("Unexpected primary predictor class %s", dir_update_ptr->dir1_class);
  }
}

unsigned char get_dir2_max(struct bpred_t* pred, struct bpred_update_t* dir_update_ptr) {
  switch (dir_update_ptr->dir1_class) {
  case BPredNbit:
    return get_2lev_max(pred);   // dir1 is bimodal, so dir2 is 2-level
  case BPred2Level:
    return get_bimod_max(pred);  // dir1 is 2-level, so dir1 is bimodal
  default:
    fatal("Unexpected primary predictor class %s", dir_update_ptr->dir1_class);
  }
}

unsigned char get_dir1_halfway(struct bpred_t* pred, struct bpred_update_t* dir_update_ptr) {
  switch (dir_update_ptr->dir1_class) {
  case BPredNbit:
    return get_bimod_halfway(pred);
  case BPred2Level:
    return get_2lev_halfway(pred);
  default:
    fatal("Unexpected primary predictor class %s", dir_update_ptr->dir1_class);
  }
}

unsigned char get_dir2_halfway(struct bpred_t* pred, struct bpred_update_t* dir_update_ptr) {
  switch (dir_update_ptr->dir1_class) {
  case BPredNbit:
    return get_2lev_halfway(pred);   // dir1 is bimodal, so dir2 is 2-level
  case BPred2Level:
    return get_bimod_halfway(pred);  // dir1 is 2-level, so dir1 is bimodal
  default:
    fatal("Unexpected primary predictor class %s", dir_update_ptr->dir1_class);
  }
}

// static
int is_stateless(struct bpred_t* pred) {
  return pred->class == BPredNotTaken || pred->class == BPredTaken || pred->class == BPredSmartStatic;
}
