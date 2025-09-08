/**
 * Copyright (C) 2025 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

import { ReactNode } from 'react';

import { Disclosure, DisclosurePanel, DisclosureTitle, Flex, Radio, RadioGroup, Text } from '@geti/ui';
import { clsx } from 'clsx';

import styles from './radio-disclosure-group.module.scss';

interface RadioDisclosureGroupProps<Value extends string> {
    ariaLabel: string;
    value: Value | null;
    onChange: (value: Value) => void;
    items: { value: Value; label: string; icon: ReactNode; content?: ReactNode }[];
}

export const RadioDisclosureGroup = <Value extends string>({
    ariaLabel,
    onChange,
    items,
    value,
}: RadioDisclosureGroupProps<Value>) => {
    return (
        <RadioGroup
            width={'100%'}
            isEmphasized
            aria-label={ariaLabel}
            onChange={(newValue) => onChange(newValue as Value)}
            value={value}
        >
            {items.map((item) => (
                <Disclosure
                    isQuiet
                    key={item.label}
                    UNSAFE_className={clsx(styles.disclosure)}
                    onExpandedChange={() => {
                        onChange(item.value);
                    }}
                    isExpanded={item.value === value && item.content !== undefined}
                >
                    <DisclosureTitle UNSAFE_className={styles.disclosureTitleContainer}>
                        <Flex alignItems={'center'} justifyContent={'space-between'} width={'100%'}>
                            <Flex alignItems={'center'} gap={'size-100'}>
                                {item.icon}
                                <Text UNSAFE_className={styles.disclosureTitle}>{item.label}</Text>
                            </Flex>
                            <Radio value={item.value} />
                        </Flex>
                    </DisclosureTitle>
                    <DisclosurePanel>{item.content}</DisclosurePanel>
                </Disclosure>
            ))}
        </RadioGroup>
    );
};
